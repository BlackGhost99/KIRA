"""Import borné et reproductible de conversations françaises OASST1.

Pas de datasets, Hub, PyArrow ou PyYAML. Les réponses sont conservées entières ;
un exemple trop long est écarté, jamais coupé pour fabriquer une réponse courte.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.request

from .conversation_eval import reserved_prompts
from .dialogue_data import curriculum, normalize, validate

REVISION = "fdf72ae0827c1cda404aff25b6603abec9e3399b"
BASE_URL = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/" + REVISION + "/"
ARCHIVE = "2023-04-12_oasst_ready.trees.jsonl.gz"
FILES = {
    ARCHIVE: (34145252, "2a9a8fd343e9b28e04a895a669d3253f82d93e9c174d440199ae19d5fafbdff7"),
    "README.md": (10168, "68483ac2fcc2f3f7779f453352363827678070ef35b6d746ccb2ca6958540fff"),
    "LICENSE": (11357, "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"),
}
MAX_UNCOMPRESSED = 256 * 1024 * 1024
MAX_LINE = 4 * 1024 * 1024


def digest_file(path):
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(chunk)
            digest.update(chunk)
    return count, digest.hexdigest()


def acquire(raw_dir):
    """Accepte aussi un cache hors ligne ; aucune désactivation du TLS."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = raw_dir / name
        if not path.exists():
            part = path.with_name(path.name + ".part")
            try:
                with urllib.request.urlopen(BASE_URL + name, timeout=45) as response, part.open("wb") as target:
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > expected[0]:
                            raise ValueError("Téléchargement supérieur à la taille publiée : " + name)
                        target.write(chunk)
                if digest_file(part) != expected:
                    raise ValueError("Empreinte de source incorrecte : " + name)
                part.replace(path)
            finally:
                part.unlink(missing_ok=True)
        if digest_file(path) != expected:
            raise ValueError("Cache de source incorrect : " + name)


def read_trees(path):
    size, count = 0, 0
    with gzip.open(path, "rb") as stream:
        while line := stream.readline(MAX_LINE + 1):
            size += len(line)
            count += 1
            if size > MAX_UNCOMPRESSED or len(line) > MAX_LINE or count > 20000:
                raise ValueError("Export OASST supérieur aux limites de lecture")
            item = json.loads(line)
            if item.get("tree_state") != "ready_for_export":
                raise ValueError("Arbre OASST non exportable dans la source épinglée")
            yield item


def label(node, key):
    value = node.get("labels", {}).get(key)
    if not isinstance(value, dict) or type(value.get("count")) is not int or value["count"] < 1:
        return None
    score = value.get("value")
    if type(score) not in (int, float) or not math.isfinite(score):
        return None
    return score


def acceptable(node, expected_role):
    if node.get("role") != expected_role or node.get("lang") != "fr":
        return False
    if node.get("deleted") is not False or node.get("synthetic") is not False or node.get("review_result") is not True:
        return False
    if type(node.get("review_count")) is not int or node["review_count"] < 1:
        return False
    text = node.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > 4096:
        return False
    # Étiquettes humaines : ce filtre ne prouve pas l'exactitude des réponses.
    if any(label(node, key) != 0 for key in ("spam", "lang_mismatch", "pii", "not_appropriate")):
        return False
    if expected_role == "assistant":
        quality = label(node, "quality")
        if node.get("rank") != 0 or quality is None or quality < 0.5 or label(node, "fails_task") != 0:
            return False
        if re.search(r"(?:je suis|je m'appelle|mon nom est|en tant qu).{0,60}(?:openassistant|chatgpt|claude)", text, re.I):
            return False
    return True


def extract(trees, encode, context):
    """Groupes par arbre et racine normalisée ; jamais de branche rejetée réintroduite."""
    rows, stats = [], Counter()
    reserved = reserved_prompts()
    # Conserver les références de l'ancien curriculum sans entraîner sur ses tests.
    authored = [r for r in curriculum() if r["grading"] == "reference_only"]
    occupied = {normalize(r["messages"][0]["content"]) for r in authored}
    seen_inputs = set()
    for tree in trees:
        stats["trees_read"] += 1
        root = tree["prompt"]
        if root.get("lang") != "fr":
            continue
        stats["french_root_trees"] += 1
        group = hashlib.sha256(normalize(root["text"]).encode()).hexdigest()
        bucket = int(group[:8], 16) % 10
        split = "validation" if bucket == 0 else "test" if bucket == 1 else "train"
        def walk(node, turns, ids):
            role = "prompter" if len(turns) % 2 == 0 else "assistant"
            if not acceptable(node, role):
                stats["rejected_branch_roots"] += 1
                return
            if role == "prompter" and normalize(node["text"]) in reserved | occupied:
                stats["reserved_or_authored_prompt_branches"] += 1
                return
            messages = turns + [{"role": "user" if role == "prompter" else role, "content": node["text"]}]
            message_ids = ids + [node["message_id"]]
            if role == "assistant":
                stats["quality_paths"] += 1
                if len(encode(messages)) > context + 1:
                    stats["over_context_paths"] += 1
                    # Les descendants seraient plus longs encore.
                    return
                key = json.dumps([normalize(t["content"]) for t in messages[:-1]], ensure_ascii=False)
                if key in seen_inputs:
                    stats["duplicate_paths"] += 1
                else:
                    seen_inputs.add(key)
                    rows.append({"id": "oasst1-" + node["message_id"], "group": "oasst1-" + group,
                                 "skill": "dialogue-fr", "split": split, "source": "OpenAssistant/oasst1@" + REVISION,
                                 "source_tree_id": tree["message_tree_id"], "source_message_ids": message_ids,
                                 "license": "Apache-2.0", "synthetic": False, "training_allowed": True,
                                 "grading": "reference_only", "messages": messages})
            for child in node.get("replies", []):
                walk(child, messages, message_ids)
        walk(root, [], [])
    for row in authored:
        row["skill"] = "dialogue-fr"  # Tirage uniforme, sans surpondérer les petites familles.
        if len(encode(row["messages"])) <= context + 1:
            rows.append(row)
            stats["authored_open_examples"] += 1
        else:
            stats["authored_over_context"] += 1
    rows.sort(key=lambda x: x["id"])
    validate(rows)
    stats["oasst_examples"] = sum(not r["synthetic"] for r in rows)
    return rows, dict(stats)


def prepare(checkpoint, raw_dir, output):
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError("Fichier de corpus déjà présent : choisir un nouveau chemin")
    acquire(raw_dir)
    from .dialogue import load_model, encode_messages
    model, codec = load_model(checkpoint)
    rows, stats = extract(read_trees(raw_dir / ARCHIVE), lambda m: encode_messages(m, codec)[0], model.cfg.context)
    raw = ("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    manifest = {"format": "kira-oasst-fr-v1", "source_url": BASE_URL, "source_revision": REVISION,
                "sources": {k: {"bytes": v[0], "sha256": v[1]} for k, v in FILES.items()},
                "license": "Apache-2.0; authored examples retain project attribution",
                "attribution": "OpenAssistant Conversations (OASST1), LAION and OpenAssistant contributors, 2023",
                "changes": "French, human-reviewed rank-0 paths; metadata quality and length filters; grouped split; authored open examples added",
                "tokenizer_sha256": hashlib.sha256(codec.serialized.encode()).hexdigest(),
                "context": model.cfg.context, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "examples": {s: sum(r["split"] == s for r in rows) for s in ("train", "validation", "test")},
                "counts": stats, "reserved_prompts_excluded": True,
                "scope": "small-filtered-development-corpus-not-a-general-assistant-guarantee"}
    with output.with_suffix(".manifest.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="Poids KIRA déterminant le tokenizer et le contexte")
    p.add_argument("--raw-dir", type=Path, required=True, help="Cache des trois sources épinglées, utilisable hors ligne")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        prepare(a.checkpoint, a.raw_dir, a.output)
    except (OSError, ValueError, RuntimeError) as exc:
        p.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
