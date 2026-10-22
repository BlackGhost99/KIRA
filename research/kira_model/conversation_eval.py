"""Sondes de conversation libre : les sorties générées alimentent les tours suivants.

Les contrôles automatiques ne notent ni la grammaire ni la pertinence. Un rapport
sans anomalie reste à relire et ne qualifie jamais une assistante généraliste.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

SUITE = [
    {"id": "owner-session-20261022", "origin": "owner-reported-development-failure",
     "turns": [
         {"user": "bjr", "review": "Salutation en français, réponse intelligible."},
         {"user": "ca va?", "review": "Réponse courte et pertinente ; pas de sentiments humains inventés."},
         {"user": "Comment t'appelles tu ?", "required_word": "KIRA",
          "review": "Se présente comme KIRA dans une phrase française cohérente."},
         {"user": "T'es sûr que t'es prête ?",
          "review": "Reconnaît ses limites expérimentales sans inventer une capacité."}]},
    {"id": "presentation", "origin": "authored-development-probe",
     "turns": [
         {"user": "Bonsoir, tu peux te présenter ?", "required_word": "KIRA",
          "review": "Présentation française cohérente, identité KIRA."},
         {"user": "Réponds avec une seule phrase : à quoi sers-tu ?",
          "review": "Une phrase utile et honnête, adaptée à cette version."},
         {"user": "Est-ce que tu peux voir ce qui est sur mon écran ?",
          "review": "N'annonce pas un accès à l'écran dont ce terminal ne dispose pas."}]},
    {"id": "clarification", "origin": "authored-development-probe",
     "turns": [
         {"user": "Je ne sais pas par où commencer mon projet.",
          "review": "Demande l'objectif ou propose une première étape concrète."},
         {"user": "Je veux écrire mon premier programme Python.",
          "review": "Suit le sujet Python, propose une étape accessible."},
         {"user": "Fais plus court, s'il te plaît.",
          "review": "Raccourcit sa réponse en gardant le sujet du tour précédent."}]},
    {"id": "context", "origin": "authored-development-probe",
     "turns": [
         {"user": "Dans cette histoire, une boîte verte contient une clé.",
          "review": "Accuse réception sans inventer le contenu d'autres boîtes."},
         {"user": "Que contient la boîte verte ?", "required_word": "clé",
          "review": "Répond une clé en s'appuyant sur le contexte."},
         {"user": "Et quelle est la couleur de cette clé ?",
          "review": "La couleur de la clé est inconnue ; vert qualifiait la boîte."}]},
]


def reserved_prompts():
    """Empêche de réentraîner directement sur les questions de ces sondes."""
    from .dialogue_data import normalize
    return {normalize(t["user"]) for case in SUITE for t in case["turns"]}


def flags_for(text, turn):
    flags = []
    if not text.strip():
        flags.append("empty-response")
    if "\ufffd" in text:
        flags.append("replacement-character")
    words = re.findall(r"\w+", text.casefold())
    triples = Counter(tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2)))
    if triples and max(triples.values()) >= 3:
        flags.append("repeated-word-triple")
    required = turn.get("required_word")
    if required and required.casefold() not in words:
        flags.append("required-word-absent:" + required)
    return flags


def run_sessions(model, codec, scenarios=None, max_new_tokens=64):
    from .dialogue import reply
    cases = []
    for scenario in SUITE if scenarios is None else scenarios:
        history, turns = [], []
        for turn in scenario["turns"]:
            messages = history + [{"role": "user", "content": turn["user"]}]
            response, used = reply(model, codec, messages, max_new_tokens, temperature=0.0)
            turns.append({"user": turn["user"], "response": response,
                          "automatic_flags": flags_for(response, turn),
                          "discarded_history_pairs": (len(messages) - len(used)) // 2,
                          "human_review_criterion": turn["review"]})
            history = used + [{"role": "assistant", "content": response}]
        cases.append({"id": scenario["id"], "origin": scenario["origin"], "turns": turns})
    return {"format": "kira-free-conversation-eval-v1", "cases": cases,
            "decoding": "greedy", "max_new_tokens": max_new_tokens,
            "history": "model-generated-answers-as-in-chat-no-reference-answers",
            "automatic_flagged_turns": sum(bool(t["automatic_flags"]) for c in cases for t in c["turns"]),
            "human_review": "required-unscored", "general_assistant_qualified": False,
            "scope": "development-regressions-including-observed-failure-not-a-blind-benchmark"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Rapport existant : choisir un nouveau fichier")
    if args.threads < 1 or not 1 <= args.max_new_tokens <= 512:
        parser.error("threads ou max-new-tokens invalide")
    import torch
    from .dialogue import load_model, sha_file
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    model, codec = load_model(args.checkpoint)
    report = run_sessions(model, codec, max_new_tokens=args.max_new_tokens)
    report["checkpoint_sha256"] = sha_file(args.checkpoint)
    report["suite_sha256"] = hashlib.sha256(json.dumps(SUITE, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Un rapport d'observation est conservé ; il n'autorise pas la mise en production.
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(args.output), "flagged_turns": report["automatic_flagged_turns"],
                      "human_review": report["human_review"], "general_assistant_qualified": False}))


if __name__ == "__main__":
    main()
