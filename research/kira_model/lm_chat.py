"""KIRA LM sur CPU : dialogue local, sondes de conversation et fichier d'inférence allégé.

    python -m research.kira_model.lm_chat chat --weights poids.pt
    python -m research.kira_model.lm_chat probe --weights poids.pt --output rapport.json
    python -m research.kira_model.lm_chat export --weights RUN/weights.pt --output kira-inference.pt

Ce terminal ne lit pas tes fichiers et ne contrôle pas l'ordinateur ; /calc utilise l'outil exact existant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .codec import Codec
from .lm import FORMAT, KiraLM, config_from_dict, generate


def load(path, device="cpu"):
    """Poids KIRA LM (weights.pt, last.pt ou fichier allégé) ; uniquement des fichiers de confiance."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != FORMAT or saved.get("kind") != "weights":
        raise ValueError(f"{path} n'est pas un fichier de poids KIRA LM")
    model = KiraLM(config_from_dict(saved["config"]))
    model.load_state_dict({k: v.float() for k, v in saved["state_dict"].items()})
    return model.to(device).eval(), Codec(saved["tokenizer_json"]), saved


def reply(model, codec, messages, max_new_tokens=64, temperature=0.0, top_k=0, top_p=1.0,
          repetition_penalty=1.0, generator=None):
    """Retire les plus anciens échanges si le contexte manque, jamais le dernier message."""
    history = list(messages)
    while True:
        prompt = codec.prompt(history)
        if len(prompt) + max_new_tokens <= model.cfg.context:
            break
        system = history[:1] if history[0]["role"] == "system" else []
        rest = history[len(system):]
        if len(rest) <= 1:
            raise ValueError("Message trop long pour le contexte de ce modèle. Raccourcis-le.")
        history = system + rest[2:]
    ids = generate(model, prompt, max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
                   repetition_penalty=repetition_penalty, stop={codec.eos}, banned=codec.banned(),
                   generator=generator)
    return codec.decode(ids).strip(), history


def chat(args):
    from .dialogue import calculate
    torch.set_num_threads(args.threads)
    model, codec, saved = load(args.weights)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    print(f"KIRA LM ({saved.get('phase', '?')}, étape {saved.get('step', '?')}) — modèle expérimental, "
          "ses réponses peuvent être fausses. /reset, /quit, /calc expression.")
    while True:
        try:
            text = input("Toi > ").strip()
        except EOFError:
            break
        command = text.casefold()
        if command == "/quit":
            break
        if command == "/reset":
            history = []
            print("Conversation effacée.")
            continue
        if command == "/calc" or command.startswith("/calc "):
            try:
                print("Outil de calcul > " + calculate(text[5:].strip()))
            except ValueError as exc:
                print(str(exc))
            continue
        if not text:
            continue
        try:
            answer, used = reply(model, codec, history + [{"role": "user", "content": text}], args.max_new_tokens,
                                 args.temperature, args.top_k, args.top_p, args.repetition_penalty, generator)
        except ValueError as exc:
            print(str(exc))
            continue
        print("KIRA > " + (answer or "[aucune réponse générée]"))
        if answer:
            history = used + [{"role": "assistant", "content": answer}]
        else:
            # Une réponse vide n'entre pas dans l'historique : le message reste sans réponse inventée.
            history = used[:-1]


def probe(args):
    from .conversation_eval import SUITE, run_sessions
    if args.output.exists():
        raise FileExistsError("Rapport existant : choisir un nouveau fichier")
    torch.set_num_threads(args.threads)
    model, codec, saved = load(args.weights)
    report = run_sessions(model, codec, max_new_tokens=args.max_new_tokens, reply_fn=reply)
    digest = hashlib.sha256()
    with open(args.weights, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    report |= {"model_format": FORMAT, "weights_sha256": digest.hexdigest(), "phase": saved.get("phase"),
               "step": saved.get("step"),
               "suite_sha256": hashlib.sha256(json.dumps(SUITE, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "flagged_turns": report["automatic_flagged_turns"],
                      "human_review": report["human_review"], "general_assistant_qualified": False}))


def export(args):
    if args.output.exists():
        raise FileExistsError("Fichier existant : choisir un nouveau nom")
    model, codec, saved = load(args.weights)
    state, copies = {}, {}
    for key, value in model.state_dict().items():
        # float16 : moitié moins lourd à télécharger ; le chargement repasse en float32 pour le CPU.
        ref = value.data_ptr()
        copies.setdefault(ref, value.detach().to(torch.float16))
        state[key] = copies[ref]
    kept = {k: saved[k] for k in ("phase", "step", "tokens_seen", "signature", "status", "provenance",
                                  "selected", "selected_step", "validation") if k in saved}
    value = {"format": FORMAT, "kind": "weights", "config": saved["config"], "tokenizer_json": codec.serialized,
             "state_dict": state, "storage": "float16", **kept}
    temp = args.output.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(args.output)
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size}))


def main():
    parser = argparse.ArgumentParser(description="KIRA LM : terminal de dialogue local, sondes et export")
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("chat")
    c.add_argument("--temperature", type=float, default=0.7)
    c.add_argument("--top-k", type=int, default=0)
    c.add_argument("--top-p", type=float, default=0.9)
    c.add_argument("--repetition-penalty", type=float, default=1.1)
    c.add_argument("--seed", type=int, default=42)
    c.set_defaults(func=chat)
    p = commands.add_parser("probe")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=probe)
    e = commands.add_parser("export")
    e.add_argument("--output", type=Path, required=True)
    e.set_defaults(func=export)
    for command in (c, p, e):
        command.add_argument("--weights", type=Path, required=True)
        command.add_argument("--threads", type=int, default=2)
        command.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    try:
        args.func(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, str(exc) + "\n")
    except KeyboardInterrupt:
        parser.exit(130, "\nArrêt demandé.\n")


if __name__ == "__main__":
    main()
