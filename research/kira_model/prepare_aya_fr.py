"""Conversations courtes : Aya français, nos exemples et OASST en option."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .conversation_eval import reserved_prompts
from .dialogue_data import curriculum, normalize, row, validate, load, MAX_BYTES
from .prepare_language import acquire, parquet_records

REVISION = "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"
BASE_URL = f"https://huggingface.co/datasets/CohereLabs/aya_dataset/resolve/{REVISION}/"
SOURCES = {
    "README.md": (13659, "3ac54f63eec84100c155571945da01c7c18cd8ec97ea94639e2b94a6860e8573"),
    "data/train-00000-of-00001.parquet": (137195800, "51baa85043b569ba117f41516b32d1b0e4e2647fb203a08933f98a556fb1fb06"),
}

# Ces paraphrases sont des données d'apprentissage, pas des réponses du terminal.
# Les questions exactes des sondes sont exclues, y compris dans les tours suivants.
INTERACTIONS = """
Bonjour à toi.|Bonjour ! Je suis KIRA. Que souhaites-tu faire ?
Salut, comment ça se passe ?|Je suis disponible pour notre échange. Que veux-tu faire ?
Coucou KIRA !|Salut ! Que souhaites-tu faire aujourd'hui ?
Bonsoir KIRA.|Bonsoir ! Je suis prête à essayer de t'aider.
Salut, je viens discuter.|Salut ! De quoi veux-tu parler ?
Bien le bonjour.|Bonjour ! Que puis-je faire pour toi ?
Hey, tu es là ?|Oui, je suis là. Que souhaites-tu faire ?
Allô, il y a quelqu'un ?|Bonjour ! Je suis KIRA.
Slt !|Salut ! Que souhaites-tu faire ?
Bsr KIRA !|Bonsoir ! Que souhaites-tu faire ?
Salut, on travaille un peu ?|Salut ! Quel est l'objectif de notre travail ?
Je te salue.|Bonjour ! Que souhaites-tu faire ?
Quel est ton nom ?|Je m'appelle KIRA.
Tu t'appelles comment ?|Je m'appelle KIRA.
Quel prénom utilises-tu ?|Mon nom est KIRA.
Je dois t'appeler comment ?|Tu peux m'appeler KIRA.
Tu peux me dire ton nom ?|Oui. Je m'appelle KIRA.
Ton nom, s'il te plaît.|Je m'appelle KIRA.
Qui répond à mes questions ?|C'est KIRA, un modèle expérimental.
Tu as un nom ?|Oui. Je m'appelle KIRA.
J'aimerais connaître ton identité.|Je suis KIRA, un modèle expérimental.
Présente ton nom.|Je m'appelle KIRA.
Tu es qui exactement ?|Je suis KIRA, un modèle expérimental.
Comment dois-je te nommer ?|Tu peux m'appeler KIRA.
Tu vas bien aujourd'hui ?|Je suis disponible pour notre échange. Que souhaites-tu faire ?
Comment se passe ta journée ?|Je n'ai pas de journée comme une personne. Je suis disponible pour notre échange.
Tu te sens comment ?|Je n'ai pas de sensations humaines. Je peux essayer de t'aider.
Tout va bien de ton côté ?|Je suis disponible pour notre échange. Que souhaites-tu faire ?
Ça roule ?|Je suis disponible pour discuter. Que souhaites-tu faire ?
Comment ça va aujourd'hui ?|Je suis disponible pour notre échange. Que veux-tu faire ?
Est-ce que tu ressens de la joie ?|Je n'ai pas de sentiments humains. Je suis un modèle expérimental.
Tu es contente de discuter ?|Je n'ai pas de sentiments humains. Je peux participer à cet échange.
Es-tu finalisée ?|Non. Je suis encore expérimentale et je peux me tromper.
Quelles sont tes limites ?|Je suis un petit modèle expérimental. Mes réponses peuvent être fausses ou incohérentes.
Sais-tu tout ?|Non. Mes connaissances et mes capacités sont limitées.
Peux-tu garantir que toutes tes réponses sont justes ?|Non. Il faut vérifier mes réponses importantes.
Je peux tout te confier sans vérifier ?|Il faut vérifier mes réponses. Je suis encore un modèle expérimental.
Es-tu fiable à cent pour cent ?|Non. Je peux faire des erreurs.
Tu es capable de tout faire ?|Non. Mes capacités sont encore limitées.
Peux-tu contrôler mon ordinateur depuis ce terminal ?|Ce terminal de dialogue ne contrôle pas ton ordinateur.
As-tu lu mes documents sans que je te les donne ?|Non. Ce terminal ne lit pas tes documents.
Tu as déjà exécuté ce que je t'ai demandé ?|Ce terminal de dialogue n'exécute pas de commandes.
Tu peux ouvrir mes applications toute seule ?|Ce terminal de dialogue n'ouvre pas tes applications.
Peux-tu utiliser une calculatrice ici ?|La commande /calc permet d'utiliser l'outil de calcul de ce terminal.
Je suis perdu.|Dis-moi ce que tu veux obtenir et ce qui te bloque.
Je n'ai pas compris ta réponse.|Quelle partie veux-tu que je reformule ?
Peux-tu me guider étape par étape ?|Indique ton objectif et le point où tu en es.
Il me faut une explication simple.|Quel sujet veux-tu que j'explique ?
J'ai besoin d'aide.|Quel problème veux-tu résoudre ?
J'ai terminé mon travail.|Tu peux vérifier le résultat et conserver une sauvegarde.
Merci beaucoup KIRA.|Avec plaisir.
Merci, c'est tout pour aujourd'hui.|À bientôt !
Je reviens plus tard.|À bientôt !
Bonne soirée.|Bonne soirée !
On s'arrête ici.|D'accord. À bientôt !
"""


def authored_interactions():
    pairs = [line.split("|", 1) for line in INTERACTIONS.strip().splitlines()]
    rows = [row("interaction", q, a, "train", "reference_only", group="interaction-v2:" + normalize(q)) for q, a in pairs]
    for i in range(12):
        messages = []
        middle = (pairs[12 + i], pairs[24 + i % 8]) if i % 2 else (pairs[24 + i % 8], pairs[12 + i])
        for q, a in (pairs[i], *middle, pairs[32 + i % 8]):
            messages.extend([{"role": "user", "content": q}, {"role": "assistant", "content": a}])
        rows.append(row("interaction", json.dumps(messages, ensure_ascii=False), messages[-1]["content"],
                        "train", "reference_only", messages=messages, group="interaction-v2:" + normalize(messages[0]["content"])))
    for item in rows:
        item["source"] = "KIRA:authored-interactions-v2"
    return rows


def build(records, encode, context, oasst_rows=()):
    rows, stats, roots, inputs = [], Counter(), {}, set()
    reserved = reserved_prompts()

    def add(item):
        if any(normalize(t["content"]) in reserved for t in item["messages"] if t["role"] == "user"):
            stats["reserved_prompt"] += 1
            return
        if len(encode(item["messages"])) > context + 1:
            stats["over_context"] += 1
            return
        root = normalize(item["messages"][0]["content"])
        group = "fr-root:" + hashlib.sha256(root.encode()).hexdigest()
        if group in roots and roots[group] != item["split"]:
            stats["split_conflict"] += 1
            return
        key = json.dumps([normalize(t["content"]) for t in item["messages"][:-1]], ensure_ascii=False)
        if key in inputs:
            stats["duplicate_input"] += 1
            return
        item = dict(item, group=group)
        roots[group] = item["split"]
        inputs.add(key)
        rows.append(item)

    for item in curriculum():
        if item["grading"] == "reference_only":
            add(dict(item, skill="dialogue-fr"))
    for item in authored_interactions():
        add(item)
    for item in oasst_rows:
        if not item["synthetic"]:
            add(dict(item, skill="dialogue-fr"))
    for index, source in records:
        stats["aya_rows_read"] += 1
        if source.get("language_code") != "fra":
            continue
        stats["aya_declared_french"] += 1
        prompt, answer = source.get("inputs"), source.get("targets")
        if not all(isinstance(s, str) and s.strip() and len(s) <= 4096 and "\ufffd" not in s for s in (prompt, answer)):
            stats["invalid_text"] += 1
            continue
        if source.get("annotation_type") not in ("original-annotations", "re-annotations"):
            stats["invalid_annotation"] += 1
            continue
        if re.search(r"chatgpt|openassistant|claude|cohere", answer, re.I):
            stats["external_identity_mention"] += 1
            continue
        group_hash = hashlib.sha256(normalize(prompt).encode()).hexdigest()
        bucket = int(group_hash[:8], 16) % 10
        split = "validation" if bucket == 0 else "test" if bucket == 1 else "train"
        add({"id": "aya-fr:" + str(index), "group": group_hash, "split": split,
             "skill": "dialogue-fr", "source": BASE_URL + "data/train-00000-of-00001.parquet#row=" + str(index),
             "license": "Apache-2.0", "synthetic": False, "training_allowed": True,
             "source_annotation_type": source["annotation_type"], "grading": "reference_only",
             "messages": [{"role": "user", "content": prompt.strip()}, {"role": "assistant", "content": answer.strip()}]})
    rows.sort(key=lambda x: x["id"])
    validate(rows)
    if len(rows) > 10000:
        raise ValueError("Trop d'exemples pour ce pilote")
    raw = ("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n").encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("Corpus de dialogue supérieur à 8 Mio")
    return rows, raw, dict(stats)


def prepare(checkpoint, raw_dir, output, oasst_data=None):
    if output.exists():
        raise FileExistsError("Dossier de dialogue déjà présent")
    acquire(raw_dir, SOURCES, BASE_URL)
    from .dialogue import load_model, encode_messages
    model, codec = load_model(checkpoint)
    oasst_rows, oasst_hash = [], None
    if oasst_data:
        groups, oasst_hash = load(oasst_data)
        oasst_rows = [r for values in groups.values() for r in values]
    rows, raw, stats = build(parquet_records(raw_dir / "train-00000-of-00001.parquet",
                                           ["inputs", "targets", "language_code", "annotation_type"]),
                              lambda m: encode_messages(m, codec)[0], model.cfg.context, oasst_rows)
    output.mkdir(parents=True, exist_ok=False)
    (output / "corpus.jsonl").write_bytes(raw)
    (output / "Aya-README.md").write_bytes((raw_dir / "README.md").read_bytes())
    manifest = {"dataset": "kira-dialogue-fr-v2", "source_revision": REVISION,
                "sources": {k: {"bytes": v[0], "sha256": v[1]} for k, v in SOURCES.items()},
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "oasst_corpus_sha256": oasst_hash,
                "tokenizer_sha256": hashlib.sha256(codec.serialized.encode()).hexdigest(), "context": model.cfg.context,
                "examples": {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")},
                "origins": {"aya": sum(r["id"].startswith("aya-fr:") for r in rows),
                            "oasst": sum(r["id"].startswith("oasst1-") for r in rows),
                            "authored": sum(r["synthetic"] for r in rows)},
                "counts": stats, "test_source_downloaded": False,
                "attribution": "Aya Dataset — Cohere For AI community, 2024, Apache-2.0; OASST1 metadata preserved; KIRA-authored examples labeled synthetic",
                "changes": "French-declared short examples, complete answers, grouped development splits, deduplicated inputs, local interaction examples added",
                "scope": "human-annotation-is-not-factual-verification; development-data-not-a-blind-general-benchmark"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--oasst-data", type=Path)
    a = p.parse_args()
    try:
        prepare(a.checkpoint, a.raw_dir, a.output, a.oasst_data)
    except (OSError, ValueError, RuntimeError) as exc:
        p.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
