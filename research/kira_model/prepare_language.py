"""Petit échantillon LeCarnet reproductible pour le laboratoire CPU.

PyArrow ne sert qu'à préparer les fichiers Parquet. Le corpus JSONL résultant est
utilisable par le PC Windows avec les dépendances KIRA déjà installées.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request

from .dialogue_data import normalize

REPO = "MaxLSB/LeCarnet"
REVISION = "d0eaae9f8f7f777fde86f16c0b607da1335d2700"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
SOURCES = {
    "README.md": (2724, "f3393599d3daeeadae3421055111425d85a6fffdc58af226a28f2520979489e1"),
    "data/train-00000-of-00004.parquet": (213040725, "b3ff2ab7966383ac6f83687ccaeb894033e57048ee6680baf27db234fbd89e6e"),
    "data/validation-00000-of-00001.parquet": (8738596, "ab9a1d62a333905c59ffe74070e7381cac90ad06efbaed5c5a1b017b96c05a4e"),
}
MAX_OUTPUT = 14 * 1024 * 1024


def digest_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def acquire(raw_dir, sources=None, base_url=None):
    sources = SOURCES if sources is None else sources
    base_url = BASE_URL if base_url is None else base_url
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in sources.items():
        path = raw_dir / Path(name).name
        if path.exists():
            if digest_file(path) != expected:
                raise ValueError("Cache LeCarnet incorrect : " + name)
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            with urllib.request.urlopen(base_url + name, timeout=45) as response, temporary.open("wb") as stream:
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > expected[0]:
                        raise ValueError("Source supérieure à sa taille publiée : " + name)
                    stream.write(chunk)
            if digest_file(temporary) != expected:
                raise ValueError("Empreinte LeCarnet incorrecte : " + name)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def parquet_records(path, columns):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("La préparation Parquet demande PyArrow ; l'entraînement du JSONL n'en a pas besoin") from exc
    # Lecture par lots : ne matérialise pas tout le shard dans la RAM du PC.
    with pq.ParquetFile(path) as source:
        index = 0
        for chunk in source.iter_batches(batch_size=256, columns=columns, use_threads=False):
            for row in chunk.to_pylist():
                yield index, row
                index += 1


def parquet_texts(path):
    for index, row in parquet_records(path, ["text"]):
        yield index, row["text"]


def select_documents(validation_rows, train_rows, train_count=5000, validation_count=64):
    if not 1 <= train_count <= 5000 or not 1 <= validation_count <= 64:
        raise ValueError("Échantillon limité à 5000 histoires et 64 validations")
    documents, hashes, prefixes, stats = [], set(), set(), Counter()
    for split, candidates, count, stride in (("validation", validation_rows, validation_count, 251),
                                             ("train", train_rows, train_count, 89)):
        kept = 0
        for index, text in candidates:
            stats[split + "_scanned"] += 1
            if index >= 600000:
                raise ValueError("Trop de lignes parcourues pour cet échantillon")
            if index % stride:
                continue
            if not isinstance(text, str) or not 200 <= len(text) <= 4096 or "\ufffd" in text:
                stats["invalid_text"] += 1
                continue
            normalized = normalize(text)
            digest = hashlib.sha256(normalized.encode()).hexdigest()
            prefix = hashlib.sha256(normalized[:160].encode()).hexdigest()
            if digest in hashes or prefix in prefixes:
                stats["duplicate_text_or_prefix"] += 1
                continue
            hashes.add(digest)
            prefixes.add(prefix)
            file = "data/validation-00000-of-00001.parquet" if split == "validation" else "data/train-00000-of-00004.parquet"
            documents.append({"id": f"lecarnet:{split}:{index}", "split": split, "text": text.strip(),
                              "source": BASE_URL + file + "#row=" + str(index), "source_row": index,
                              "license": "ODC-By-1.0", "training_allowed": True, "synthetic": True,
                              "attribution": "LeCarnet — MaxLSB et jddqd; source texts generated with Mistral-Large-Instruct-2411"})
            kept += 1
            if kept == count:
                break
        if kept != count:
            raise ValueError(f"Échantillon {split} incomplet : {kept}/{count}")
    documents.sort(key=lambda x: x["id"])
    raw = ("\n".join(json.dumps(d, ensure_ascii=False, sort_keys=True) for d in documents) + "\n").encode()
    if len(raw) > MAX_OUTPUT:
        raise ValueError("Échantillon supérieur à 14 Mio")
    return raw, dict(stats)


def prepare(raw_dir, output, train_count=5000, validation_count=64):
    if output.exists():
        raise FileExistsError("Dossier de corpus déjà présent : choisir un nouveau dossier")
    acquire(raw_dir)
    raw, stats = select_documents(parquet_texts(raw_dir / "validation-00000-of-00001.parquet"),
                                  parquet_texts(raw_dir / "train-00000-of-00004.parquet"), train_count, validation_count)
    output.mkdir(parents=True, exist_ok=False)
    (output / "corpus.jsonl").write_bytes(raw)
    (output / "LeCarnet-README.md").write_bytes((raw_dir / "README.md").read_bytes())
    manifest = {"dataset": "kira-lecarnet-small-v1", "source_repo": REPO, "source_revision": REVISION,
                "sources": {k: {"bytes": v[0], "sha256": v[1]} for k, v in SOURCES.items()},
                "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "examples": {"train": train_count, "validation": validation_count, "test": 0},
                "selection": "first-train-shard-stride-89; official-validation-stride-251; fixed-count; exact-and-prefix-dedup",
                "license": "ODC-By-1.0 (publisher declaration)",
                "license_url": "https://opendatacommons.org/licenses/by/1-0/",
                "attribution": "LeCarnet — MaxLSB et jddqd; generated with Mistral-Large-Instruct-2411",
                "changes": "bounded subset; exact-normalized and 160-character-prefix deduplication; original train/validation preserved",
                "test_source_downloaded": False, "test_source_evaluated": False, "counts": stats,
                "scope": "simple-synthetic-French-stories-not-general-knowledge-or-dialogue-data"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-count", type=int, default=5000)
    p.add_argument("--validation-count", type=int, default=64)
    a = p.parse_args()
    try:
        prepare(a.raw_dir, a.output, a.train_count, a.validation_count)
    except (OSError, ValueError) as exc:
        p.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
