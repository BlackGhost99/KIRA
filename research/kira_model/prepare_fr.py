"""Corpus littéraire FR reproductible, quatre téléchargements bornés et identifiés."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import textwrap
import unicodedata
import urllib.request

from .data import MAX_CORPUS_BYTES, read_corpus

SOURCES = (
    {"id": 800, "author": "Jules Verne", "title": "Le tour du monde en quatre-vingts jours",
     "split": "train", "sha256": "bd89ef4c406ae063a1eeba2ec0689532e8c83ecfb1fce742cdd22bc47cb888da"},
    {"id": 4650, "author": "Voltaire", "title": "Candide, ou l'optimisme",
     "split": "train", "sha256": "1cc10ab3268a76aa01bd3f9722ae2ae0e5e36b869a18b0bb0832b75bc0327806"},
    {"id": 11714, "author": "Guy de Maupassant", "title": "Contes de la Bécasse",
     "split": "validation", "sha256": "d78ea5ac752a2f3572c153d0c974dbaf62b01c25b7ca8994f6a2b6fc9158b0eb"},
    {"id": 12065, "author": "Gustave Flaubert", "title": "Trois contes",
     "split": "test", "sha256": "7f82586956f4a8f4185ef7ff056692660f46dce70038a1380b6d9ffba38d63b4"},
)
SOURCE_LIMIT = 2 * 1024 * 1024
LICENSE = "Project Gutenberg eBook terms; catalog status: public domain in the USA"


def clean_body(raw: bytes) -> str:
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    start = re.search(r"^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*.*$", text, re.M)
    end = re.search(r"^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*.*$", text, re.M)
    if not start or not end or start.end() >= end.start():
        raise ValueError("Bornes Gutenberg absentes ou désordonnées : nettoyage refusé")
    return unicodedata.normalize("NFC", text[start.end():end.start()]).strip()


def chunks(text: str, size: int = 2000) -> list[str]:
    """Découpage sans chevauchement ; conserve tous les mots du corps."""
    result, current = [], ""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        for part in textwrap.wrap(paragraph, width=size, break_long_words=False, break_on_hyphens=False):
            if current and len(current) + len(part) + 2 > size:
                result.append(current)
                current = ""
            current = current + "\n\n" + part if current else part
    if current:
        result.append(current)
    return result


def source_bytes(spec: dict, raw_dir: Path) -> bytes:
    destination = raw_dir / f"{spec['id']}.txt"
    if destination.exists():
        with destination.open("rb") as handle:
            raw = handle.read(SOURCE_LIMIT + 1)
    else:
        url = f"https://www.gutenberg.org/cache/epub/{spec['id']}/pg{spec['id']}.txt"
        request = urllib.request.Request(url, headers={"User-Agent": "KIRA-Seed-research/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read(SOURCE_LIMIT + 1)
    if len(raw) > SOURCE_LIMIT or hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError(f"Source {spec['id']} différente de la version vérifiée : import refusé")
    if not destination.exists():
        temp = destination.with_suffix(".tmp")
        temp.write_bytes(raw)
        temp.replace(destination)
    return raw


def build_rows(raw_by_id: dict[int, bytes], sources=SOURCES):
    rows, metadata, seen = [], [], set()
    for spec in sources:
        pieces = chunks(clean_body(raw_by_id[spec["id"]]))
        # Évaluation courte et déterministe, répartie sur tout un ouvrage indépendant.
        count = len(pieces) if spec["split"] == "train" else min(8, len(pieces))
        indices = list(range(len(pieces))) if count == len(pieces) else [
            i * (len(pieces) - 1) // (count - 1) for i in range(count)]
        kept = []
        for index in indices:
            text = pieces[index]
            digest = hashlib.sha256(" ".join(unicodedata.normalize("NFKC", text).casefold().split()).encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            kept.append(index)
            rows.append({"id": f"pg-{spec['id']}-{index:04d}", "split": spec["split"],
                         "source": f"https://www.gutenberg.org/ebooks/{spec['id']}#sha256={spec['sha256']}",
                         "license": LICENSE, "training_allowed": True, "text": text})
        metadata.append({**spec, "catalog": f"https://www.gutenberg.org/ebooks/{spec['id']}",
                         "raw_bytes": len(raw_by_id[spec["id"]]), "available_chunks": len(pieces),
                         "selected_chunks": kept, "terms": "https://www.gutenberg.org/policy/license.html"})
    return rows, metadata


def prepare(output: Path):
    if (output / "corpus.jsonl").exists() or (output / "sources.json").exists():
        raise FileExistsError("Corpus déjà présent : choisir un nouveau dossier pour ne pas l'écraser")
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = {spec["id"]: source_bytes(spec, raw_dir) for spec in SOURCES}
    rows, sources = build_rows(raw)
    encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    if len(encoded) > MAX_CORPUS_BYTES:
        raise ValueError("Corpus préparé trop volumineux")
    temp = output / "corpus.tmp"
    temp.write_bytes(encoded)
    docs, digest = read_corpus(temp)
    summary = {"name": "kira-fr-literature-v1", "corpus_sha256": digest,
               "bytes": len(encoded), "sources": sources,
               "counts": {key: len(value) for key, value in docs.items()},
               "token_counts": {key: sum(len(d['tokens']) for d in value) for key, value in docs.items()},
               "limitations": "Historical fiction only; no dialogue tuning; exact-chunk deduplication only; evaluation samples eight chunks from each held-out book."}
    temp.replace(output / "corpus.jsonl")
    (output / "sources.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"corpus": str(output / "corpus.jsonl"), "bytes": len(encoded),
                      "counts": summary["counts"], "token_counts": summary["token_counts"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prépare quatre ouvrages français vérifiés ; aucun poids de modèle téléchargé")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare(args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Préparation arrêtée : {exc}\nLes sources déjà vérifiées restent dans raw pour une nouvelle tentative.\n")


if __name__ == "__main__":
    main()
