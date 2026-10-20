"""python -m research.kira_model {train,generate} --help"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from .data import BOS, EOS, TOKENIZER, decode, encode, read_corpus
from .model import Config, KiraSeed


def batch(windows, context):
    xs, ys = [], []
    for tokens in windows:
        x, y = tokens[:context], tokens[1:context + 1]
        # La dernière entrée sans cible doit être ignorée ; padding toujours à droite.
        valid = len(y)
        xs.append(x[:valid] + [0] * (context - valid))
        ys.append(y + [-100] * (context - valid))
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def evaluate(model, docs, context):
    model.eval()
    total, count = 0.0, 0
    for doc in docs:
        tokens = doc["tokens"]
        for start in range(0, len(tokens) - 1, context):
            x, y = batch([tokens[start:start + context + 1]], context)
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="sum")
            total += loss.item()
            count += int((y != -100).sum())
    return total / count


def train(args):
    if args.steps < 1 or args.batch_size < 1 or args.threads < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("steps, batch-size, threads et lr doivent être positifs")
    cfg = Config(width=args.width, heads=args.heads, layers=args.layers, context=args.context)
    docs, corpus_hash = read_corpus(args.corpus)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = KiraSeed(cfg)  # Aucun from_pretrained, poids externes ou appel de modèle tiers.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.output.mkdir(parents=True, exist_ok=False)  # Un nouveau run, jamais d'écrasement.
    initial = evaluate(model, docs["validation"], cfg.context)
    history = []
    seen = 0
    for step in range(1, args.steps + 1):
        windows = []
        for _ in range(args.batch_size):
            tokens = rng.choice(docs["train"])["tokens"]
            start = rng.randrange(max(1, len(tokens) - cfg.context))
            windows.append(tokens[start:start + cfg.context + 1])
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
    final = evaluate(model, docs["validation"], cfg.context)
    manifest = {
        "format": 1, "model": "kira-seed-0.1", "status": "experimental-not-an-assistant",
        "initialization": "random", "tokenizer": TOKENIZER, "config": asdict(cfg),
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch_version": str(torch.__version__), "device": "cpu", "seed": args.seed,
        "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.lr,
        "tokens_seen": seen, "corpus_sha256": corpus_hash,
        "documents": {split: [{k: v for k, v in d.items() if k != "tokens"} for d in rows]
                      for split, rows in docs.items()},
        "validation_loss_initial": initial, "validation_loss_final": final,
        "history": history,
    }
    temp = args.output / "weights.tmp"
    torch.save({"config": asdict(cfg), "tokenizer": TOKENIZER, "state_dict": model.state_dict()}, temp)
    temp.replace(args.output / "weights.pt")
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"parameters": manifest["parameters"], "validation_initial": initial,
                      "validation_final": final, "output": str(args.output)}), flush=True)


def generate(args):
    if not 1 <= args.max_new_tokens <= 4096 or args.threads < 1 or not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("Génération : limites invalides")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    # Charger seulement un checkpoint local de confiance ; weights_only n'est pas un bac à sable.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["tokenizer"] != TOKENIZER:
        raise ValueError("Tokenizer incompatible")
    model = KiraSeed(Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    tokens = encode(args.prompt, end=False)
    if len(tokens) > model.cfg.context:
        raise ValueError("Prompt trop long pour le contexte en octets de ce prototype")
    continuation = []
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            x = torch.tensor([tokens[-model.cfg.context:]], dtype=torch.long)
            logits = model(x)[0, -1].clone()
            logits[BOS] = float("-inf")
            if args.temperature == 0:
                token = int(logits.argmax())
            else:
                token = int(torch.multinomial(F.softmax(logits / args.temperature, dim=-1), 1))
            if token == EOS:
                break
            tokens.append(token)
            continuation.append(token)
    print(decode(continuation))


def main():
    parser = argparse.ArgumentParser(description="KIRA Seed : laboratoire CPU, sans connexion à la production")
    commands = parser.add_subparsers(dest="command", required=True)
    t = commands.add_parser("train")
    t.add_argument("--corpus", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    for name, default in (("steps", 200), ("batch-size", 4), ("width", 64), ("heads", 4),
                          ("layers", 2), ("context", 128), ("seed", 42), ("threads", 2)):
        t.add_argument("--" + name, type=int, default=default)
    t.add_argument("--lr", type=float, default=0.001)
    t.set_defaults(func=train)
    g = commands.add_parser("generate")
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-new-tokens", type=int, default=100)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--threads", type=int, default=2)
    g.set_defaults(func=generate)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileExistsError) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
