"""Expérience BPE locale, indépendante des checkpoints par octets de KIRA Seed."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

# Le tokenizer fonctionne sur CPU et lit seulement les documents d'entraînement.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import tokenizers
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import torch
from torch import nn
from torch.nn import functional as F

from .__main__ import batch, evaluate
from .data import decode as decode_bytes, read_corpus
from .model import Config, KiraSeed

FORMAT = "kira-seed-bpe-0.2"
CODEC = "bytelevel-bpe-virtual-boundaries-v1"


class TextCodec:
    def __init__(self, serialized: str):
        self.backend = Tokenizer.from_str(serialized)
        self.serialized = self.backend.to_str()
        self.base_size = self.backend.get_vocab_size()
        self.bos, self.eos = self.base_size, self.base_size + 1
        self.vocab_size = self.base_size + 2

    def encode(self, text: str, end: bool = True) -> list[int]:
        return [self.bos, *self.backend.encode(text, add_special_tokens=False).ids,
                *([self.eos] if end else [])]

    def decode(self, ids: list[int]) -> str:
        return self.backend.decode([i for i in ids if 0 <= i < self.base_size],
                                   skip_special_tokens=False)


def fit_codec(documents: dict, vocab_size: int) -> TextCodec:
    if not 258 <= vocab_size <= 8192:
        raise ValueError("vocab-size doit être compris entre 258 et 8192")
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size - 2, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  max_token_length=32, show_progress=False)
    backend.train_from_iterator((decode_bytes(d["tokens"]) for d in documents["train"]),
                                trainer=trainer, length=len(documents["train"]))
    # BOS/EOS sont des identifiants virtuels : aucun texte littéral n'est consommé comme balise.
    return TextCodec(backend.to_str())


def recode(documents: dict, codec: TextCodec) -> dict:
    result = {}
    for split, rows in documents.items():
        result[split] = []
        for row in rows:
            text = decode_bytes(row["tokens"])
            result[split].append({**row, "tokens": codec.encode(text),
                                  "utf8_bytes": len(text.encode("utf-8"))})
    return result


class BPEModel(KiraSeed):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__(cfg)
        self.token = nn.Embedding(vocab_size, cfg.width)
        self.head = nn.Linear(cfg.width, vocab_size, bias=False)
        self._init(self.token)
        self.head.weight = self.token.weight


def code_hash() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in ("bpe_experiment.py", "model.py", "data.py", "__main__.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def atomic_save(value: dict, path: Path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def report_validation(model, rows: list, context: int) -> dict:
    loss = evaluate(model, rows, context)
    predicted = sum(len(d["tokens"]) - 1 for d in rows)
    byte_count = sum(d["utf8_bytes"] for d in rows)
    return {"loss_per_token": loss, "bits_per_utf8_byte": loss * predicted / byte_count / math.log(2),
            "predicted_tokens": predicted, "utf8_bytes": byte_count}


def train(args):
    if args.steps < 1 or args.batch_size < 1 or args.threads < 1 or args.checkpoint_every < 1:
        raise ValueError("steps, batch-size, threads et checkpoint-every doivent être positifs")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr doit être positif et fini")
    if not 258 <= args.vocab_size <= 8192:
        raise ValueError("vocab-size doit être compris entre 258 et 8192")
    cfg = Config(width=args.width, heads=args.heads, layers=args.layers, context=args.context)
    original, corpus_hash = read_corpus(args.corpus)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    saved = None
    if args.resume:
        saved = torch.load(args.output / "training.pt", map_location="cpu", weights_only=True)
        if saved.get("format") != FORMAT:
            raise ValueError("Checkpoint incompatible : cette expérience attend un checkpoint BPE")
        codec = TextCodec(saved["tokenizer_json"])
    else:
        if args.output.exists():
            raise FileExistsError("Dossier de run déjà présent ; utiliser --resume ou un nouveau dossier")
        codec = fit_codec(original, args.vocab_size)
    signature = {"config": asdict(cfg), "corpus_sha256": corpus_hash, "codec": CODEC,
                 "tokenizer_sha256": hashlib.sha256(codec.serialized.encode()).hexdigest(),
                 "requested_vocab_size": args.vocab_size, "seed": args.seed, "threads": args.threads,
                 "batch_size": args.batch_size, "lr": args.lr, "torch": str(torch.__version__),
                 "tokenizers": str(tokenizers.__version__), "code_sha256": code_hash()}
    if saved and saved["signature"] != signature:
        raise ValueError("Reprise incompatible : corpus, tokenizer, options, code ou bibliothèques différents")
    docs = recode(original, codec)
    model = BPEModel(cfg, codec.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if saved:
        start = saved["step"]
        if args.steps <= start:
            raise ValueError("steps doit dépasser le nombre d'étapes enregistré")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        initial, history, seen = saved["initial"], saved["history"], saved["tokens_seen"]
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        initial = report_validation(model, docs["validation"], cfg.context)
        start, history, seen = 0, [], 0

    def save_training(step):
        atomic_save({"format": FORMAT, "signature": signature, "config": asdict(cfg),
                     "tokenizer_json": codec.serialized, "step": step,
                     "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                     "initial": initial, "history": history, "tokens_seen": seen},
                    args.output / "training.pt")

    if not saved:
        save_training(0)
        (args.output / "tokenizer.json").write_text(codec.serialized, encoding="utf-8")
    stats = {}
    for split, rows in docs.items():
        stats[split] = {"documents": len(rows), "utf8_bytes": sum(d["utf8_bytes"] for d in rows),
                        "tokens": sum(len(d["tokens"]) for d in rows)}
    print(json.dumps({"model": FORMAT, "vocab_size": codec.vocab_size,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "corpus": stats, "resume_step": start}), flush=True)
    began = time.perf_counter()
    for step in range(start + 1, args.steps + 1):
        windows = []
        for _ in range(args.batch_size):
            ids = rng.choice(docs["train"])["tokens"]
            offset = rng.randrange(max(1, len(ids) - cfg.context))
            windows.append(ids[offset:offset + cfg.context + 1])
        x, y = batch(windows, cfg.context)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError("Perte non finie : entraînement arrêté")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        seen += int((y != -100).sum())
        if step == 1 or step % 50 == 0 or step == args.steps:
            history.append({"step": step, "train_loss": loss.item()})
            print(json.dumps(history[-1]), flush=True)
        if step % args.checkpoint_every == 0 or step == args.steps:
            save_training(step)
    seconds = time.perf_counter() - began
    final = report_validation(model, docs["validation"], cfg.context)
    manifest = {"format": FORMAT, "status": "experimental-not-an-assistant",
                "initialization": "random", "config": asdict(cfg), "signature": signature,
                "vocab_size": codec.vocab_size, "parameters": sum(p.numel() for p in model.parameters()),
                "steps": args.steps, "resumed_from_step": start, "tokens_seen": seen,
                "seconds_per_step": seconds / (args.steps - start), "tokenization": stats,
                "validation_initial": initial, "validation_final": final,
                "test_split_evaluated": False, "history": history}
    atomic_save({"format": FORMAT, "config": asdict(cfg), "tokenizer_json": codec.serialized,
                 "state_dict": model.state_dict()}, args.output / "weights.pt")
    temp = args.output / "manifest.tmp"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(args.output / "manifest.json")
    print(json.dumps({"steps": args.steps, "parameters": manifest["parameters"],
                      "validation": final, "seconds_per_step": manifest["seconds_per_step"],
                      "output": str(args.output)}), flush=True)


def generate(args):
    if args.threads < 1 or not 1 <= args.max_new_tokens <= 4096:
        raise ValueError("Limites de génération invalides")
    if args.top_k < 0 or not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("temperature ou top-k invalide")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if saved.get("format") != FORMAT:
        raise ValueError("Checkpoint incompatible : génération BPE attendue")
    codec = TextCodec(saved["tokenizer_json"])
    model = BPEModel(Config(**saved["config"]), codec.vocab_size)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    ids = codec.encode(args.prompt, end=False)
    if len(ids) > model.cfg.context:
        raise ValueError("Prompt trop long pour le contexte en sous-mots")
    continuation = []
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            logits = model(torch.tensor([ids[-model.cfg.context:]], dtype=torch.long))[0, -1].clone()
            logits[codec.bos] = float("-inf")
            if args.temperature == 0:
                token = int(logits.argmax())
            else:
                if args.top_k:
                    threshold = torch.topk(logits, min(args.top_k, codec.vocab_size - 1)).values[-1]
                    logits[logits < threshold] = float("-inf")
                token = int(torch.multinomial(F.softmax(logits / args.temperature, dim=-1), 1))
            if token == codec.eos:
                break
            ids.append(token)
            continuation.append(token)
    print(codec.decode(continuation))


def main():
    parser = argparse.ArgumentParser(description="KIRA Seed BPE : vocabulaire et poids appris depuis zéro sur CPU")
    commands = parser.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train")
    t.add_argument("--corpus", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    for name, default in (("steps", 1000), ("batch-size", 1), ("threads", 1), ("width", 128),
                          ("layers", 4), ("heads", 4), ("context", 256), ("vocab-size", 2048),
                          ("seed", 42), ("checkpoint-every", 25)):
        t.add_argument("--" + name, type=int, default=default)
    t.add_argument("--lr", type=float, default=0.001)
    t.add_argument("--resume", action="store_true")
    t.set_defaults(func=train)
    g = commands.add_parser("generate")
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--prompt", required=True)
    for name, default in (("threads", 1), ("seed", 42), ("max-new-tokens", 100), ("top-k", 40)):
        g.add_argument("--" + name, type=int, default=default)
    g.add_argument("--temperature", type=float, default=0.5)
    g.set_defaults(func=generate)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")
    except KeyboardInterrupt:
        parser.exit(130, "Arrêt demandé. Le dernier training.pt complet permet une reprise avec --resume.\n")


if __name__ == "__main__":
    main()
