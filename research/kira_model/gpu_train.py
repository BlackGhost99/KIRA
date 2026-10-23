"""Entraînement GPU de KIRA LM, reprenable d'une session Kaggle à l'autre.

Préentraînement, poids initialisés de zéro :
    python -m research.kira_model.gpu_train pretrain --data DONNEES --output RUN --preset kira-100m
    torchrun --standalone --nproc_per_node 2 -m research.kira_model.gpu_train pretrain ...   # deux GPU
Dialogue, depuis nos propres poids :
    python -m research.kira_model.gpu_train sft --base RUN/weights.pt --conversations conv.jsonl \
        --replay-data DONNEES --output RUN-DIALOGUE --steps 3000

Une session s'arrête d'elle-même avant --time-limit et laisse training.pt. La suivante repart de ce
fichier (--resume-from) ; les données, le modèle et les réglages d'apprentissage doivent être identiques.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import platform
import subprocess
import time
import warnings

import torch
from torch import distributed as dist
import tokenizers

from .codec import Codec, check_turns
from .lm import FORMAT, KiraLM, PRESETS, config_from_dict, generate, parameter_count, preset, token_losses

TRAINING_FORMAT = "kira-lm-training-0.5"
DATA_FORMAT = "kira-token-shards-1"
CODE_FILES = ("gpu_train.py", "lm.py", "codec.py", "lm_chat.py")
SAMPLE_PROMPTS = ("Il était une fois", "La physique étudie", "Pour apprendre une nouvelle langue,",
                  "Bonjour ! Je m'appelle")
SHAPES = ("1-sqrt", "linear", "cosine")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_sha():
    digest = hashlib.sha256()
    for name in CODE_FILES:
        digest.update(name.encode())
        # Fins de ligne normalisées : même empreinte pour une copie Windows ou Linux du même code.
        digest.update((Path(__file__).parent / name).read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def code_commit():
    if os.environ.get("KIRA_CODE_COMMIT"):
        return os.environ["KIRA_CODE_COMMIT"]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def image_commit():
    """Version de l'image Docker de Kaggle (fichier /etc/git_commit), absente ailleurs."""
    marker = Path("/etc/git_commit")
    return marker.read_text(encoding="utf-8", errors="replace").strip()[:80] if marker.exists() else None


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_save(value, path):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def finite(value):
    """JSON strict : une valeur infinie ou NaN devient null dans les journaux."""
    return value if math.isfinite(value) else None


def generator_for(*parts):
    """Tirage déterminé par (graine, étape, micro-lot global) : une reprise retrouve les mêmes lots."""
    seed = int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little")
    return torch.Generator().manual_seed(seed & (2**63 - 1))


def cpu_state(module):
    """Copie CPU du state_dict qui garde un seul exemplaire des poids liés."""
    copies, state = {}, {}
    for key, value in module.state_dict().items():
        ref = (value.data_ptr(), value.dtype, tuple(value.shape))
        if ref not in copies:
            copies[ref] = value.detach().to("cpu", copy=True)
        state[key] = copies[ref]
    return state


class Runtime:
    """Processus seul ou lancé par torchrun ; NCCL sur GPU, Gloo sur CPU (tests)."""

    def __init__(self, device="auto", threads=0, precision="auto"):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local = int(os.environ.get("LOCAL_RANK", "0"))
        self.main = self.rank == 0
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("GPU demandé, mais aucun GPU CUDA n'est visible (activer l'accélérateur Kaggle)")
        cuda = device in ("auto", "cuda") and torch.cuda.is_available()
        if threads:
            torch.set_num_threads(threads)
        if cuda:
            torch.cuda.set_device(self.local)
            self.device = torch.device("cuda", self.local)
            major, minor = torch.cuda.get_device_capability(self.device)
            self.device_name, self.capability = torch.cuda.get_device_name(self.device), f"{major}.{minor}"
            # T4 (7.5) et P100 (6.0) n'ont pas de bfloat16 matériel : float16 avec GradScaler.
            native = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.device, self.device_name, self.capability, native = torch.device("cpu"), platform.machine() or "cpu", None, None
        chosen = {"auto": native, "fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
        if chosen is not None and not cuda:
            raise ValueError("float16/bfloat16 mixtes demandent un GPU ; utiliser --precision fp32 sur CPU")
        if chosen == torch.bfloat16 and native != torch.bfloat16:
            raise ValueError("Ce GPU n'a pas de bfloat16 matériel : utiliser fp16 ou fp32")
        self.amp = chosen
        if self.world > 1:
            # Délai large : le processus 0 sauvegarde et génère des exemples pendant que les autres attendent.
            dist.init_process_group(backend="nccl" if cuda else "gloo", timeout=timedelta(minutes=30))

    @property
    def precision(self):
        return {None: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[self.amp]

    def autocast(self):
        if self.amp is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp)

    def sum(self, values):
        tensor = torch.tensor([float(v) for v in values], dtype=torch.float64, device=self.device)
        if self.world > 1:
            dist.all_reduce(tensor)
        return tensor.tolist()

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def close(self):
        if self.world > 1 and dist.is_initialized():
            dist.destroy_process_group()


class Shard:
    def __init__(self, path, tokens, sha256):
        self.path, self.tokens, self.sha256, self._data = Path(path), int(tokens), sha256, None

    def data(self):
        if self._data is None:
            if self.path.stat().st_size != 2 * self.tokens:
                raise ValueError(f"Fichier de jetons incomplet : {self.path}")
            with open(self.path, "rb") as stream:
                self._map = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # tampon en lecture seule : on ne l'écrit jamais
                self._data = torch.frombuffer(self._map, dtype=torch.uint16)
        return self._data

    def window(self, offset, length):
        return self.data()[offset:offset + length].to(torch.long)


class TokenData:
    """Jetons préparés par corpus.py : un dossier par source et par split, uint16 petit-boutiste."""

    def __init__(self, root):
        self.root = Path(root)
        raw = (self.root / "manifest.json").read_bytes()
        self.manifest = json.loads(raw)
        if self.manifest.get("format") != DATA_FORMAT or self.manifest.get("dtype") != "uint16":
            raise ValueError("Dossier de données incompatible : préparer les jetons avec corpus.py")
        info = self.manifest["tokenizer"]
        self.codec = Codec((self.root / info["file"]).read_text(encoding="utf-8"))
        if self.codec.sha256 != info["sha256"]:
            raise ValueError("Le tokenizer du dossier ne correspond pas au manifeste")
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.shards = {}
        for name, source in self.manifest["sources"].items():
            for split, part in source["splits"].items():
                self.shards[(name, split)] = [Shard(self.root / s["file"], s["tokens"], s["sha256"])
                                              for s in part["shards"]]

    def verify(self, splits=("train", "validation"), full=True):
        for (name, split), shards in self.shards.items():
            for shard in shards if split in splits else ():
                if shard.path.stat().st_size != 2 * shard.tokens or (full and sha_file(shard.path) != shard.sha256):
                    raise ValueError(f"Fichier de jetons différent du manifeste : {shard.path}")

    def mix(self, requested, context):
        weights = requested or {name: source.get("default_weight", 0)
                                for name, source in self.manifest["sources"].items()}
        mix = {}
        for name, weight in sorted(weights.items()):
            if name not in self.manifest["sources"]:
                raise ValueError("Source inconnue dans le mélange : " + name)
            if not (math.isfinite(weight) and weight >= 0):
                raise ValueError("Poids de mélange invalide pour " + name)
            if weight and not any(s.tokens > context + 1 for s in self.shards.get((name, "train"), [])):
                raise ValueError("Source sans assez de jetons d'entraînement : " + name)
            if weight:
                mix[name] = weight
        total = sum(mix.values())
        if not total:
            raise ValueError("Le mélange de sources est vide")
        return {name: weight / total for name, weight in mix.items()}

    def sample(self, mix, rows, context, generator):
        names = list(mix)
        probabilities = torch.tensor([mix[n] for n in names], dtype=torch.float64)
        windows = []
        for _ in range(rows):
            name = names[int(torch.multinomial(probabilities, 1, generator=generator))]
            shards = [s for s in self.shards[(name, "train")] if s.tokens > context + 1]
            sizes = torch.tensor([s.tokens for s in shards], dtype=torch.float64)
            shard = shards[int(torch.multinomial(sizes, 1, generator=generator))]
            offset = int(torch.randint(0, shard.tokens - context, (1,), generator=generator))
            windows.append(shard.window(offset, context + 1))
        batch = torch.stack(windows)
        return batch[:, :-1], batch[:, 1:]

    def windows(self, name, split, context, max_tokens):
        """Fenêtres fixes, sans chevauchement, au début des fichiers : la même mesure à chaque validation."""
        found, budget = [], max_tokens
        for index, shard in enumerate(self.shards.get((name, split), [])):
            offset = 0
            while offset + context + 1 <= shard.tokens and budget >= context:
                found.append((index, offset))
                offset += context
                budget -= context
        return found

    def bytes_per_token(self, name, split):
        part = self.manifest["sources"][name]["splits"].get(split, {})
        return part["utf8_bytes"] / part["tokens"] if part.get("tokens") else None

    def provenance(self):
        keys = ("repo", "revision", "license", "license_url", "attribution", "origin", "files")
        return {name: {k: source[k] for k in keys if k in source} for name, source in self.manifest["sources"].items()}


def load_conversations(path, codec, context):
    """Conversations JSONL préparées par sft_data.py. Le split test n'est ni encodé ni évalué ici."""
    from .conversation_eval import reserved_prompts
    from .dialogue_data import normalize
    raw = Path(path).read_bytes()
    reserved = reserved_prompts()
    groups, ids, group_split = {"train": [], "validation": [], "test": []}, set(), {}
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ("id", "group", "source", "license"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Conversation ligne {line_no} : champ {field} requis")
        if row.get("training_allowed") is not True or row.get("split") not in groups:
            raise ValueError(f"Conversation ligne {line_no} : autorisation ou split invalide")
        check_turns(row.get("messages"))
        if row["messages"][-1]["role"] != "assistant":
            raise ValueError(f"Conversation ligne {line_no} : elle doit finir par une réponse")
        if row["id"] in ids or group_split.setdefault(row["group"], row["split"]) != row["split"]:
            raise ValueError(f"Conversation ligne {line_no} : identifiant répété ou groupe sur deux splits")
        if any(t["role"] == "user" and normalize(t["content"]) in reserved for t in row["messages"]):
            raise ValueError(f"Conversation ligne {line_no} : question réservée aux sondes d'évaluation")
        ids.add(row["id"])
        groups[row["split"]].append(row)
    if not groups["train"] or not groups["validation"]:
        raise ValueError("Conversations d'entraînement et de validation requises")
    encoded = {}
    for split in ("train", "validation"):
        encoded[split] = []
        for row in groups[split]:
            tokens, learn = codec.conversation(row["messages"])
            if len(tokens) > context + 1:
                raise ValueError(f"Conversation {row['id']} : {len(tokens)} jetons pour un contexte de {context}")
            encoded[split].append({"id": row["id"], "ids": tokens, "learn": learn})
    counts = {split: len(rows) for split, rows in groups.items()}
    sources = sorted({(r["source"], r["license"]) for rows in groups.values() for r in rows})
    return encoded, hashlib.sha256(raw).hexdigest(), counts, [{"source": s, "license": l} for s, l in sources]


def conversation_batch(items, pad):
    length = max(len(item["ids"]) - 1 for item in items)
    xs, ys = [], []
    for item in items:
        targets = [t if keep else -100 for t, keep in zip(item["ids"][1:], item["learn"][1:])]
        extra = length - len(targets)
        xs.append(item["ids"][:-1] + [pad] * extra)
        ys.append(targets + [-100] * extra)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def learning_rate(step, peak, warmup, plan):
    """Montée linéaire, plateau, puis décroissance finale (WSD) quand un plan est fixé."""
    if step <= warmup:
        return peak * step / max(1, warmup)
    if not plan or step <= plan["start"]:
        return peak
    progress = min(1.0, (step - plan["start"]) / plan["steps"])
    factor = {"1-sqrt": 1 - math.sqrt(progress), "linear": 1 - progress,
              "cosine": 0.5 * (1 + math.cos(math.pi * progress))}[plan["shape"]]
    return peak * (plan["min_ratio"] + (1 - plan["min_ratio"]) * factor)


def make_optimizer(model, args, device):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay}, {"params": other, "weight_decay": 0.0}]
    extra = {"fused": True} if device.type == "cuda" else {}
    return torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, **extra)


def parse_mix(text):
    if not text:
        return None
    mix = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        try:
            mix[name.strip()] = float(value)
        except ValueError as exc:
            raise ValueError("Mélange attendu sous la forme source=poids,source=poids") from exc
    return mix


@torch.no_grad()
def token_validation(raw, runtime, data, names, context, max_tokens, rows):
    raw.eval()
    result = {}
    for name in names:
        mine = data.windows(name, "validation", context, max_tokens)[runtime.rank::runtime.world]
        shards = data.shards.get((name, "validation"), [])
        total = count = 0.0
        for i in range(0, len(mine), rows):
            batch = torch.stack([shards[s].window(o, context + 1) for s, o in mine[i:i + rows]]).to(runtime.device)
            with runtime.autocast():
                logits = raw(batch[:, :-1])
            loss, _, n = token_losses(logits, batch[:, 1:])
            total, count = total + loss.item(), count + n.item()
        total, count = runtime.sum([total, count])
        if count:
            loss = total / count
            ratio = data.bytes_per_token(name, "validation")
            result[name] = {"loss_per_token": loss, "tokens": int(count),
                            "approx_bits_per_utf8_byte": loss / math.log(2) / ratio if ratio else None}
    raw.train()
    return result


@torch.no_grad()
def conversation_validation(raw, runtime, items, pad, rows):
    raw.eval()
    total = count = 0.0
    mine = items[runtime.rank::runtime.world]
    for i in range(0, len(mine), rows):
        x, y = conversation_batch(mine[i:i + rows], pad)
        with runtime.autocast():
            logits = raw(x.to(runtime.device))
        loss, _, n = token_losses(logits, y.to(runtime.device))
        total, count = total + loss.item(), count + n.item()
    total, count = runtime.sum([total, count])
    raw.train()
    return {"assistant_loss_per_token": total / count, "assistant_tokens": int(count)}


@torch.no_grad()
def text_samples(raw, runtime, codec, max_new_tokens=48):
    raw.eval()
    samples = []
    with runtime.autocast():
        for i, prompt in enumerate(SAMPLE_PROMPTS):
            ids = [codec.bos, *codec.encode(prompt)]
            room = min(max_new_tokens, raw.cfg.context - len(ids))
            if room < 1:
                continue
            greedy = generate(raw, ids, room, stop={codec.eos}, banned=codec.banned())
            sampled = generate(raw, ids, room, temperature=0.8, top_p=0.95, stop={codec.eos}, banned=codec.banned(),
                               generator=torch.Generator(device=runtime.device).manual_seed(1000 + i))
            samples.append({"prompt": prompt, "greedy": codec.decode(greedy), "sampled": codec.decode(sampled)})
    raw.train()
    return samples


@torch.no_grad()
def conversation_samples(raw, runtime, codec):
    from .conversation_eval import run_sessions
    from .lm_chat import reply

    def observed_reply(model, codec, messages, max_new_tokens, temperature=0.0):
        # Une sonde trop longue pour le contexte est un échec à consigner, pas une raison d'arrêter l'entraînement.
        try:
            return reply(model, codec, messages, max_new_tokens, temperature)
        except ValueError as exc:
            return f"[non généré : {exc}]", messages

    raw.eval()
    with runtime.autocast():
        report = run_sessions(raw, codec, max_new_tokens=min(64, raw.cfg.context // 4), reply_fn=observed_reply)
    raw.train()
    return [{"id": case["id"], "turns": [{"user": t["user"], "response": t["response"], "flags": t["automatic_flags"]}
                                         for t in case["turns"]]} for case in report["cases"]]


def check_args(args, phase):
    for name in ("micro_batch", "eval_every", "eval_tokens", "log_every", "max_skipped_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} doit être positif")
    for name in ("lr", "weight_decay", "beta1", "beta2", "eps", "grad_clip", "z_loss", "time_limit",
                 "checkpoint_minutes", "final_reserve"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name.replace('_', '-')} doit être positif ou nul et fini")
    if args.lr == 0 or not 0 <= args.beta1 < 1 or not 0 <= args.beta2 < 1:
        raise ValueError("lr doit être positif et les betas compris entre 0 et 1")
    if args.warmup < 0 or (args.session_steps is not None and args.session_steps < 1):
        raise ValueError("warmup et session-steps doivent être positifs")
    if not 0 <= args.min_lr_ratio <= 1 or args.decay_shape not in SHAPES:
        raise ValueError("min-lr-ratio doit être entre 0 et 1, decay-shape parmi " + ", ".join(SHAPES))
    if phase == "pretrain":
        if args.batch_tokens < 1 or (args.max_steps is not None and args.max_steps < 1):
            raise ValueError("batch-tokens et max-steps doivent être positifs")
        if (args.decay_start is None) != (args.decay_steps is None):
            raise ValueError("decay-start et decay-steps vont ensemble")
        if args.decay_start is not None and (args.decay_start < args.warmup or args.decay_steps < 1):
            raise ValueError("La décroissance commence après le warmup et dure au moins une étape")
        if not (args.max_steps or args.decay_start is not None or args.time_limit or args.session_steps):
            raise ValueError("Fixer une fin : max-steps, decay-start/decay-steps, time-limit ou session-steps")
    else:
        if args.steps < 1 or args.batch_size < 1 or not args.warmup < args.steps:
            raise ValueError("steps et batch-size doivent être positifs, warmup inférieur à steps")
        if not 0 <= args.replay_ratio < 1 or args.replay_rows < 1 or args.max_language_regression < 0:
            raise ValueError("replay-ratio dans [0, 1[, replay-rows positif, max-language-regression positif")


def load_training(path, phase):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != TRAINING_FORMAT or saved.get("kind") != "training" or saved.get("phase") != phase:
        raise ValueError(f"{path} n'est pas un état d'entraînement KIRA LM de phase {phase}")
    return saved


def load_weights(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != FORMAT or saved.get("kind") != "weights":
        raise ValueError(f"{path} n'est pas un fichier de poids KIRA LM")
    return saved


def build_signature(args, phase, cfg, codec, extra):
    # Tout ce qui change ce que le modèle apprend. Le plan de décroissance du préentraînement reste à part :
    # il se décide en fin de parcours et se fige dès qu'il a commencé.
    return {"phase": phase, "config": asdict(cfg), "tokenizer_sha256": codec.sha256, "seed": args.seed,
            "lr": args.lr, "warmup": args.warmup, "weight_decay": args.weight_decay,
            "betas": [args.beta1, args.beta2], "eps": args.eps, "grad_clip": args.grad_clip,
            "z_loss": args.z_loss, **extra}


def train(args, phase):
    runtime = Runtime(args.device, args.threads, args.precision)
    try:
        return _train(args, phase, runtime)
    finally:
        runtime.close()


def _train(args, phase, runtime):
    began = time.monotonic()
    check_args(args, phase)
    output = Path(args.output)
    origin = None
    if (output / "training.pt").exists():
        origin = output
    elif args.resume_from:
        origin = Path(args.resume_from)
        if not (origin / "training.pt").exists():
            raise FileNotFoundError(f"Aucun training.pt dans {origin}")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("Dossier de run non vide et sans training.pt : choisir un nouveau dossier")
    saved = load_training(origin / "training.pt", phase) if origin else None

    data = replay = conversations = None
    if phase == "pretrain":
        data = TokenData(args.data)
        if runtime.main:
            # Empreintes complètes pour un nouveau run ; tailles seulement lors d'une reprise, sauf --verify-data.
            data.verify(full=args.verify_data or saved is None)
        codec = data.codec
        overrides = {k: getattr(args, k) for k in ("width", "layers", "heads", "kv_heads", "hidden", "context")
                     if getattr(args, k) is not None}
        # Recalculée à chaque session : une reprise avec une autre forme est refusée par la signature.
        cfg = preset(args.preset, codec.vocab_size, **overrides)
        mix = data.mix(parse_mix(args.mix), cfg.context)
        rows = args.micro_batch * cfg.context * runtime.world
        if args.batch_tokens % rows:
            raise ValueError(f"batch-tokens doit être un multiple de micro-batch × context × GPU = {rows}")
        accumulation = args.batch_tokens // rows
        extra = {"data_sha256": data.sha256, "mix": mix, "batch_tokens": args.batch_tokens}
        base_state, provenance = None, {"pretraining": data.provenance()}
    else:
        if saved:
            codec, cfg, base_state = Codec(saved["tokenizer_json"]), config_from_dict(saved["config"]), None
            base_sha = saved["signature"]["base_sha256"]
            if args.base and sha_file(args.base) != base_sha:
                raise ValueError("Reprise incompatible : poids de départ différents")
        else:
            if not args.base:
                raise ValueError("--base doit désigner des poids KIRA LM préentraînés")
            base = load_weights(args.base)
            codec, cfg, base_state, base_sha = Codec(base["tokenizer_json"]), config_from_dict(base["config"]), base["state_dict"], sha_file(args.base)
        conversations, conversations_sha, counts, sources = load_conversations(args.conversations, codec, cfg.context)
        mix = None
        if args.replay_data:
            replay = TokenData(args.replay_data)
            if replay.codec.sha256 != codec.sha256:
                raise ValueError("Les données de répétition n'utilisent pas le tokenizer de ces poids")
            mix = replay.mix(parse_mix(args.replay_mix), cfg.context)
        ratio = args.replay_ratio if replay else 0.0
        if args.batch_size % (args.micro_batch * runtime.world):
            raise ValueError("batch-size doit être un multiple de micro-batch × nombre de GPU")
        accumulation = args.batch_size // (args.micro_batch * runtime.world)
        extra = {"base_sha256": base_sha, "conversations_sha256": conversations_sha, "batch_size": args.batch_size,
                 "steps": args.steps, "replay_sha256": replay.sha256 if replay else None, "replay_ratio": ratio,
                 "replay_mix": mix, "replay_rows": args.replay_rows,
                 "max_language_regression": args.max_language_regression,
                 "schedule": {"shape": args.decay_shape, "min_ratio": args.min_lr_ratio}}
        provenance = {"conversations": {"examples": counts, "sources": sources},
                      "replay": replay.provenance() if replay else None}

    signature = build_signature(args, phase, cfg, codec, extra)
    session = {"started_at": now_iso(), "start_step": saved["step"] if saved else 0, "world_size": runtime.world,
               "device": runtime.device_name, "capability": runtime.capability, "precision": runtime.precision,
               "torch": str(torch.__version__), "tokenizers": str(tokenizers.__version__),
               "python": platform.python_version(), "code_sha256": code_sha(), "code_commit": code_commit(),
               "micro_batch": args.micro_batch, "accumulation": accumulation, "compile": bool(args.compile),
               "platform_image": image_commit()}
    if saved:
        differences = sorted(k for k in signature.keys() | saved["signature"].keys()
                             if signature.get(k) != saved["signature"].get(k))
        if differences:
            raise ValueError("Reprise incompatible, options ou données différentes : " + ", ".join(differences))
        if saved["sessions"][-1]["code_sha256"] != session["code_sha256"]:
            if not args.allow_code_change:
                raise ValueError("Le code d'entraînement a changé depuis la session précédente : reprendre avec le "
                                 "même commit, ou ajouter --allow-code-change (le changement sera consigné)")
            session["code_change_accepted"] = True

    if phase == "pretrain":
        plan = saved.get("decay") if saved else None
        requested = None if args.decay_start is None else {
            "start": args.decay_start, "steps": args.decay_steps, "shape": args.decay_shape, "min_ratio": args.min_lr_ratio}
        current = saved["step"] if saved else 0
        if requested and requested != plan:
            if plan and current > plan["start"]:
                raise ValueError("La décroissance a déjà commencé avec un autre plan : il ne peut plus changer")
            if requested["start"] < current:
                raise ValueError(f"decay-start doit être au moins l'étape actuelle ({current})")
            plan = requested
        end_step = plan["start"] + plan["steps"] if plan else args.max_steps
    else:
        plan = {"start": args.warmup, "steps": args.steps - args.warmup, "shape": args.decay_shape,
                "min_ratio": args.min_lr_ratio}
        end_step = args.steps

    torch.manual_seed(args.seed)
    raw = KiraLM(cfg)
    if saved or base_state is not None:
        raw.load_state_dict(saved["state_dict"] if saved else base_state)
    raw.to(runtime.device)
    optimizer = make_optimizer(raw, args, runtime.device)
    scaler = torch.amp.GradScaler("cuda") if runtime.amp == torch.float16 else None
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        if scaler and saved.get("scaler"):
            scaler.load_state_dict(saved["scaler"])
    model = raw
    if runtime.world > 1:
        model = torch.nn.parallel.DistributedDataParallel(raw, device_ids=[runtime.local] if runtime.device.type == "cuda" else None)
    if args.compile:
        model = torch.compile(model)

    def evaluate():
        if phase == "pretrain":
            return {"language": token_validation(raw, runtime, data, list(mix), cfg.context, args.eval_tokens, args.micro_batch)}
        result = {"assistant": conversation_validation(raw, runtime, conversations["validation"], codec.pad, args.micro_batch)}
        if replay:
            result["language"] = token_validation(raw, runtime, replay, list(mix), cfg.context, args.eval_tokens, args.replay_rows)
        return result

    def observe():
        if not runtime.main:
            return None
        return text_samples(raw, runtime, codec) if phase == "pretrain" else conversation_samples(raw, runtime, codec)

    if saved:
        step, tokens_seen, history, samples = saved["step"], saved["tokens_seen"], saved["history"], saved["samples"]
        sessions, elapsed_before, skipped = saved["sessions"], saved["elapsed_seconds"], saved["skipped_steps"]
        initial, last_eval, best = saved["initial_eval"], saved["last_eval"], saved["best"]
        best_state = saved.get("best_state_dict")
    else:
        step, tokens_seen, history, samples, sessions, elapsed_before, skipped = 0, 0, [], [], [], 0.0, 0
        initial = last_eval = evaluate()
        first = observe()
        if runtime.main:
            samples.append({"step": 0, "samples": first})
        best = {"step": 0, "eval": initial} if phase == "sft" else None
        best_state = cpu_state(raw) if phase == "sft" and runtime.main else None
    last_eval_step = step if not saved else saved.get("last_eval_step", -1)

    def guard_ok(evaluation):
        if not replay:
            return True
        return all(evaluation["language"][n]["loss_per_token"] <=
                   initial["language"][n]["loss_per_token"] + args.max_language_regression for n in initial["language"])

    def checkpoint(stop_reason=None):
        if not runtime.main:
            return
        output.mkdir(parents=True, exist_ok=True)
        session_now = {**session, "end_step": step, "seconds": time.monotonic() - began,
                       "ended_at": now_iso(), "stop_reason": stop_reason}
        elapsed = elapsed_before + time.monotonic() - began
        atomic_save({"format": TRAINING_FORMAT, "kind": "training", "phase": phase, "signature": signature,
                     "config": asdict(cfg), "tokenizer_json": codec.serialized, "step": step,
                     "tokens_seen": tokens_seen, "state_dict": cpu_state(raw), "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict() if scaler else None, "decay": plan if phase == "pretrain" else None,
                     "history": history, "samples": samples, "sessions": sessions + [session_now],
                     "elapsed_seconds": elapsed, "skipped_steps": skipped, "initial_eval": initial,
                     "last_eval": last_eval, "last_eval_step": last_eval_step, "best": best,
                     "best_state_dict": best_state}, output / "training.pt")
        write_json(output / "manifest.json", {
            "format": FORMAT, "phase": phase, "status": "experimental-not-a-general-assistant",
            "initialization": "random-from-scratch" if phase == "pretrain" else "continued-from-own-kira-lm-weights",
            "promotion_status": "not-qualified-requires-human-review", "parameters": parameter_count(raw),
            "config": asdict(cfg), "signature": signature, "decay_plan": plan, "steps": step,
            "planned_end_step": end_step, "tokens_seen": tokens_seen, "elapsed_seconds": elapsed,
            "skipped_steps": skipped, "initial_eval": initial, "last_eval": last_eval, "best": best,
            "test_split_evaluated": False, "provenance": provenance, "sessions": sessions + [session_now],
            "samples": samples, "history": history})

    if runtime.main:
        print(json.dumps({"phase": phase, "parameters": parameter_count(raw), "resume_step": step,
                          "end_step": end_step, "world_size": runtime.world, "accumulation": accumulation,
                          "precision": runtime.precision, "device": runtime.device_name}), flush=True)
    last_save, save_seconds, step_seconds, consecutive = time.monotonic(), 60.0, 0.0, 0
    stop_reason = None
    try:
        while True:
            elapsed = time.monotonic() - began
            reason = None
            if end_step is not None and step >= end_step:
                reason = "planned_end"
            elif args.session_steps and step >= session["start_step"] + args.session_steps:
                reason = "session_steps"
            elif args.time_limit and elapsed + 1.5 * step_seconds + 2 * save_seconds + args.final_reserve > args.time_limit:
                reason = "time_limit"
            due = args.checkpoint_minutes and time.monotonic() - last_save > 60 * args.checkpoint_minutes
            # Décision commune : tous les processus s'arrêtent ou sauvegardent à la même étape.
            stop_flag, save_flag = runtime.sum([reason is not None, bool(due)])
            if stop_flag:
                stop_reason = reason or "other_process"
                break
            if save_flag:
                started_save = time.monotonic()
                checkpoint()
                runtime.barrier()
                save_seconds = max(save_seconds, time.monotonic() - started_save)
                last_save = time.monotonic()

            step += 1
            started_step = time.monotonic()
            lr = learning_rate(step, args.lr, args.warmup, plan)
            for group in optimizer.param_groups:
                group["lr"] = lr
            objective = "language"
            # Micro-lot numéroté dans le lot global : 1 GPU × 2A ou 2 GPU × A voient les mêmes données.
            first = runtime.rank * accumulation
            if phase == "pretrain":
                batches = [data.sample(mix, args.micro_batch, cfg.context, generator_for(args.seed, step, first + m))
                           for m in range(accumulation)]
            elif replay and float(torch.rand((), generator=generator_for(args.seed, step, "replay"))) < args.replay_ratio:
                batches = [replay.sample(mix, args.replay_rows, cfg.context, generator_for(args.seed, step, first + m))
                           for m in range(accumulation)]
            else:
                objective = "assistant"
                train_items = conversations["train"]
                batches = []
                for m in range(accumulation):
                    picks = torch.randint(0, len(train_items), (args.micro_batch,),
                                          generator=generator_for(args.seed, step, first + m))
                    batches.append(conversation_batch([train_items[int(i)] for i in picks], codec.pad))
            local = sum(int((y != -100).sum()) for _, y in batches)
            (count,) = runtime.sum([local])
            if not count:
                raise RuntimeError("Lot sans cible d'apprentissage")
            ce_total = z_total = 0.0
            for m, (x, y) in enumerate(batches):
                x, y = x.to(runtime.device, non_blocking=True), y.to(runtime.device, non_blocking=True)
                last = runtime.world == 1 or m == len(batches) - 1
                with contextlib.nullcontext() if last else model.no_sync():
                    with runtime.autocast():
                        logits = model(x)
                    ce, z, _ = token_losses(logits, y, with_z=args.z_loss > 0)
                    # Moyenne sur toutes les cibles de l'étape ; DDP moyenne ensuite entre GPU, d'où × world.
                    loss = (ce + args.z_loss * z) * (runtime.world / count)
                    (scaler.scale(loss) if scaler else loss).backward()
                ce_total, z_total = ce_total + ce.detach(), z_total + z.detach()
            if scaler:
                scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), args.grad_clip)
            if scaler:
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                was_skipped = scaler.get_scale() < scale
            else:
                was_skipped = not bool(torch.isfinite(norm))
                if not was_skipped:
                    optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            skipped += was_skipped
            consecutive = consecutive + 1 if was_skipped else 0
            tokens_seen += int(count)
            step_seconds = time.monotonic() - started_step
            if consecutive >= args.max_skipped_steps:
                stop_reason = "non_finite_gradients"
                raise RuntimeError(f"{consecutive} étapes de suite avec des gradients non finis : arrêt et sauvegarde")
            if step == 1 or step % args.log_every == 0:
                ce_sum, z_sum = runtime.sum([float(ce_total), float(z_total)])
                entry = {"step": step, "objective": objective, "train_loss": finite(ce_sum / count), "lr": lr,
                         "grad_norm": finite(float(norm)), "skipped_steps": skipped, "tokens_seen": tokens_seen,
                         "step_seconds": round(step_seconds, 4), "tokens_per_second": round(count / step_seconds),
                         "elapsed_hours": round((elapsed_before + time.monotonic() - began) / 3600, 4)}
                if args.z_loss:
                    entry["z_mean"] = finite(z_sum / count)
                if scaler:
                    entry["loss_scale"] = scaler.get_scale()
                if runtime.main:
                    history.append(entry)
                    print(json.dumps(entry), flush=True)
            if step % args.eval_every == 0 or step == end_step:
                last_eval, last_eval_step = evaluate(), step
                observed = observe()
                entry = {"step": step, "validation": last_eval}
                if phase == "sft":
                    accepted = (last_eval["assistant"]["assistant_loss_per_token"] <
                                best["eval"]["assistant"]["assistant_loss_per_token"] and guard_ok(last_eval))
                    if accepted:
                        best = {"step": step, "eval": last_eval}
                        if runtime.main:
                            best_state = cpu_state(raw)
                    entry |= {"selected": accepted, "language_guard_passed": guard_ok(last_eval)}
                if runtime.main:
                    history.append(entry)
                    samples.append({"step": step, "samples": observed})
                    print(json.dumps(entry, ensure_ascii=False), flush=True)
                runtime.barrier()
    except KeyboardInterrupt:
        if runtime.world > 1:
            raise
        stop_reason = "interrupted"
    except RuntimeError:
        if stop_reason == "non_finite_gradients":
            checkpoint(stop_reason)
        raise
    if last_eval_step != step:
        last_eval, last_eval_step = evaluate(), step
        observed = observe()
        if runtime.main:
            history.append({"step": step, "validation": last_eval, "end_of_session": True})
            samples.append({"step": step, "samples": observed})
    checkpoint(stop_reason)
    if runtime.main:
        meta = {"phase": phase, "step": step, "tokens_seen": tokens_seen, "signature": signature,
                "status": "experimental-not-a-general-assistant", "provenance": provenance}
        if phase == "pretrain":
            export = {"state_dict": cpu_state(raw), "selected": "last", "validation": last_eval}
            write_weights(output / "weights.pt", cfg, codec, export | meta)
        else:
            write_weights(output / "weights.pt", cfg, codec,
                          {"state_dict": best_state, "selected": "best", "selected_step": best["step"],
                           "validation": best["eval"]} | meta)
            write_weights(output / "last.pt", cfg, codec,
                          {"state_dict": cpu_state(raw), "selected": "last", "validation": last_eval} | meta)
        print(json.dumps({"stop_reason": stop_reason, "step": step, "end_step": end_step, "tokens_seen": tokens_seen,
                          "output": str(output)}), flush=True)
    runtime.barrier()
    return {"step": step, "stop_reason": stop_reason, "tokens_seen": tokens_seen}


def write_weights(path, cfg, codec, values):
    atomic_save({"format": FORMAT, "kind": "weights", "config": asdict(cfg), "tokenizer_json": codec.serialized,
                 **values}, path)


def common_options(parser):
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, help="Dossier d'un run précédent contenant training.pt (lecture seule)")
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--z-loss", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--decay-shape", choices=SHAPES, default="1-sqrt")
    parser.add_argument("--min-lr-ratio", type=float, default=0.0)
    parser.add_argument("--time-limit", type=float, default=0.0, help="Durée maximale de la session en secondes (0 : aucune)")
    parser.add_argument("--final-reserve", type=float, default=600.0, help="Secondes gardées pour la mesure et la sauvegarde finales")
    parser.add_argument("--checkpoint-minutes", type=float, default=30.0)
    parser.add_argument("--session-steps", type=int, help="Arrêter cette session après N étapes (mesure de vitesse)")
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-tokens", type=int, default=262144, help="Jetons de validation par source")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-skipped-steps", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--compile", action="store_true", help="torch.compile (GPU de capacité 7.0 ou plus)")
    parser.add_argument("--allow-code-change", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(description="KIRA LM : entraînement GPU reprenable, poids initialisés de zéro")
    commands = parser.add_subparsers(dest="phase", required=True)
    p = commands.add_parser("pretrain")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--preset", choices=sorted(PRESETS), default="kira-100m")
    for name in ("width", "layers", "heads", "kv-heads", "hidden", "context"):
        p.add_argument("--" + name, type=int)
    p.add_argument("--batch-tokens", type=int, default=524288)
    p.add_argument("--lr", type=float, default=0.0015)
    p.add_argument("--mix", help="source=poids,source=poids (défaut : poids du manifeste)")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--decay-start", type=int, help="Étape où commence la décroissance finale du taux")
    p.add_argument("--decay-steps", type=int)
    p.add_argument("--verify-data", action="store_true", help="Recalculer les empreintes des jetons même lors d'une reprise")
    common_options(p)
    s = commands.add_parser("sft")
    s.add_argument("--base", type=Path)
    s.add_argument("--conversations", type=Path, required=True)
    s.add_argument("--replay-data", type=Path)
    s.add_argument("--replay-mix")
    s.add_argument("--replay-ratio", type=float, default=0.2)
    s.add_argument("--replay-rows", type=int, default=2)
    s.add_argument("--max-language-regression", type=float, default=0.1)
    s.add_argument("--steps", type=int, default=3000)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--lr", type=float, default=0.0003)
    common_options(s)
    s.set_defaults(warmup=100, eval_every=200, eval_tokens=65536, decay_shape="cosine", min_lr_ratio=0.1)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        train(args, args.phase)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
