"""Poursuivre le préentraînement de nos poids BPE sur un nouveau corpus français.

Le tokenizer et l'architecture restent fixes. Cette phase a son propre optimiseur,
calendrier et checkpoint ; elle ne contourne pas les vérifications d'un ancien run.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
import tokenizers

from .bpe_experiment import BPEModel, TextCodec, FORMAT as BASE_FORMAT, atomic_save, recode, report_validation
from .__main__ import batch
from .data import read_corpus
from .model import Config

FORMAT = "kira-language-phase-0.4"


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_sha():
    digest = hashlib.sha256()
    for name in ("language_phase.py", "bpe_experiment.py", "model.py", "data.py", "__main__.py"):
        digest.update(name.encode())
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


def learning_rate(step, steps, warmup, peak):
    if step <= warmup:
        return peak * step / max(1, warmup)
    phase = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * (1 + math.cos(math.pi * phase)) / 2)


def load_base(path):
    value = torch.load(path, weights_only=True, map_location="cpu")
    if value.get("format") != BASE_FORMAT:
        raise ValueError("Cette phase attend les poids BPE KIRA, avant adaptation au dialogue")
    codec = TextCodec(value["tokenizer_json"])
    model = BPEModel(Config(**value["config"]), codec.vocab_size)
    model.load_state_dict(value["state_dict"], strict=True)
    return model, codec


def select_candidate(loss, old_loss, initial_old_loss, best_loss, max_regression):
    return (math.isfinite(loss) and loss < best_loss and
            (old_loss is None or (math.isfinite(old_loss) and old_loss <= initial_old_loss + max_regression)))


def train(args):
    for key in ("steps", "threads", "batch_size", "eval_every", "checkpoint_every"):
        if getattr(args, key) < 1:
            raise ValueError("Étapes, threads, lots et intervalles doivent être positifs")
    if not 0 <= args.warmup < args.steps or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Warmup ou taux d'apprentissage invalide")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        raise ValueError("stop-after doit être compris entre 1 et steps")
    if not math.isfinite(args.replay_ratio) or not 0 <= args.replay_ratio < 1:
        raise ValueError("replay-ratio doit être compris entre 0 inclus et 1 exclu")
    if not math.isfinite(args.max_replay_regression) or args.max_replay_regression < 0:
        raise ValueError("max-replay-regression doit être positif ou nul")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    original, corpus_hash = read_corpus(args.corpus)
    saved = None
    if args.resume:
        saved = torch.load(args.output / "training.pt", weights_only=True, map_location="cpu")
        if saved.get("format") != FORMAT:
            raise ValueError("Checkpoint de phase linguistique incompatible")
        codec = TextCodec(saved["tokenizer_json"])
        model = BPEModel(Config(**saved["config"]), codec.vocab_size)
        model.load_state_dict(saved["state_dict"], strict=True)
        base_hash = saved["signature"]["base_sha256"]
        if args.base is not None and sha_file(args.base) != base_hash:
            raise ValueError("Reprise incompatible : poids de départ différents")
    else:
        if args.output.exists():
            raise FileExistsError("Dossier déjà présent : utiliser --resume ou un nouveau dossier")
        if args.base is None:
            raise ValueError("--base doit désigner les poids BPE KIRA")
        base_hash = sha_file(args.base)
        model, codec = load_base(args.base)
    documents = recode({s: original[s] for s in ("train", "validation")}, codec)
    replay, replay_hash = None, None
    if args.replay_corpus:
        previous, replay_hash = read_corpus(args.replay_corpus)
        replay = recode({s: previous[s] for s in ("train", "validation")}, codec)
    ratio = args.replay_ratio if replay else 0.0
    signature = {"base_sha256": base_hash, "corpus_sha256": corpus_hash,
                 "replay_sha256": replay_hash, "replay_ratio": ratio,
                 "max_replay_regression": args.max_replay_regression,
                 "config": asdict(model.cfg), "tokenizer_sha256": hashlib.sha256(codec.serialized.encode()).hexdigest(),
                 "steps": args.steps, "lr": args.lr, "warmup": args.warmup,
                 "batch_size": args.batch_size, "threads": args.threads, "seed": args.seed,
                 "eval_every": args.eval_every, "code_sha256": code_sha(),
                 "torch": str(torch.__version__), "tokenizers": str(tokenizers.__version__)}
    if saved and saved["signature"] != signature:
        raise ValueError("Reprise incompatible : données, code, versions ou options différents")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        rng.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        start, best_step, best_loss = saved["step"], saved["best_step"], saved["best_loss"]
        best_state, initial, history = saved["best_state_dict"], saved["initial"], saved["history"]
        replay_initial, best_replay = saved["replay_initial"], saved["best_replay"]
        seen, replay_seen = saved["new_tokens_seen"], saved["replay_tokens_seen"]
        elapsed_before = saved["elapsed_seconds"]
    else:
        initial = report_validation(model, documents["validation"], model.cfg.context)
        replay_initial = report_validation(model, replay["validation"], model.cfg.context) if replay else None
        best_state, best_step, best_loss = copy.deepcopy(model.state_dict()), 0, initial["loss_per_token"]
        best_replay = replay_initial
        start, seen, replay_seen, history, elapsed_before = 0, 0, 0, [], 0.0
    finish = args.stop_after or args.steps
    if finish <= start:
        raise ValueError("Le segment doit dépasser l'étape enregistrée")
    if not saved:
        args.output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()

    def save(step):
        atomic_save({"format": FORMAT, "signature": signature, "config": asdict(model.cfg),
                     "tokenizer_json": codec.serialized, "step": step, "state_dict": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "python_rng": rng.getstate(), "torch_rng": torch.get_rng_state(),
                     "initial": initial, "replay_initial": replay_initial, "best_replay": best_replay,
                     "best_loss": best_loss, "best_step": best_step, "best_state_dict": best_state,
                     "new_tokens_seen": seen, "replay_tokens_seen": replay_seen, "history": history,
                     "elapsed_seconds": elapsed_before + time.perf_counter() - began}, args.output / "training.pt")
    if not saved:
        save(0)
    print(json.dumps({"phase": FORMAT, "resume_step": start, "planned_steps": args.steps,
                      "parameters": sum(p.numel() for p in model.parameters()), "initial": initial,
                      "replay_initial": replay_initial}), flush=True)
    for step in range(start + 1, finish + 1):
        use_replay = replay is not None and rng.random() < ratio
        source = replay if use_replay else documents
        windows = []
        for _ in range(args.batch_size):
            ids = rng.choice(source["train"])["tokens"]
            offset = rng.randrange(max(1, len(ids) - model.cfg.context))
            windows.append(ids[offset:offset + model.cfg.context + 1])
        x, y = batch(windows, model.cfg.context)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError("Perte non finie : arrêt de la phase")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = learning_rate(step, args.steps, args.warmup, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        count = int((y != -100).sum())
        if use_replay:
            replay_seen += count
        else:
            seen += count
        if step == 1 or step % 100 == 0 or step == finish:
            print(json.dumps({"step": step, "train_loss": loss.item(), "lr": lr, "replay": use_replay}), flush=True)
        # Un arrêt de profilage ne crée pas une validation de sélection supplémentaire.
        if step % args.eval_every == 0 or step == args.steps:
            current = report_validation(model, documents["validation"], model.cfg.context)
            old = report_validation(model, replay["validation"], model.cfg.context) if replay else None
            accepted = select_candidate(current["loss_per_token"], old["loss_per_token"] if old else None,
                                        replay_initial["loss_per_token"] if replay_initial else None,
                                        best_loss, args.max_replay_regression)
            if accepted:
                best_state, best_step, best_loss, best_replay = copy.deepcopy(model.state_dict()), step, current["loss_per_token"], old
            history.append({"step": step, "validation": current, "replay_validation": old, "selected": accepted})
            print(json.dumps(history[-1]), flush=True)
        if step % args.checkpoint_every == 0 or step == finish:
            save(step)
    elapsed = elapsed_before + time.perf_counter() - began
    for name, state, selected in (("weights.pt", best_state, best_step), ("last.pt", model.state_dict(), finish)):
        atomic_save({"format": BASE_FORMAT, "config": asdict(model.cfg), "tokenizer_json": codec.serialized,
                     "state_dict": state, "language_phase_signature": signature, "selected_step": selected,
                     "status": "experimental-not-an-assistant"}, args.output / name)
    manifest = {"format": FORMAT, "status": "experimental-not-an-assistant", "signature": signature,
                "steps": finish, "planned_steps": args.steps, "best_step": best_step,
                "parameters": sum(p.numel() for p in model.parameters()), "validation_initial": initial,
                "best_validation_loss": best_loss, "replay_initial": replay_initial, "best_replay": best_replay,
                "new_tokens_seen": seen, "replay_tokens_seen": replay_seen, "elapsed_seconds": elapsed,
                "seconds_per_step": elapsed / finish, "test_split_evaluated": False,
                "tokenization": {s: {"documents": len(r), "tokens": sum(len(d["tokens"]) for d in r)} for s, r in documents.items()},
                "history": history}
    temp = args.output / "manifest.tmp"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(args.output / "manifest.json")
    print(json.dumps({"steps": finish, "best_step": best_step, "best_validation_loss": best_loss,
                      "seconds_per_step": elapsed / finish, "output": str(args.output)}), flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--replay-corpus", type=Path)
    p.add_argument("--output", type=Path, required=True)
    for name, default in (("steps", 10000), ("threads", 1), ("batch-size", 1), ("seed", 42),
                          ("warmup", 100), ("eval-every", 500), ("checkpoint-every", 100)):
        p.add_argument("--" + name, type=int, default=default)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--replay-ratio", type=float, default=0.25)
    p.add_argument("--max-replay-regression", type=float, default=0.35)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    try:
        train(args)
    except (OSError, ValueError, RuntimeError) as exc:
        p.exit(2, str(exc) + "\n")
    except KeyboardInterrupt:
        p.exit(130, "Arrêt demandé. Reprise avec --resume depuis le dernier checkpoint complet.\n")


if __name__ == "__main__":
    main()
