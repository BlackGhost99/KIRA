"""Corpus explicite, provenance et séparation avant entraînement. Aucune collecte réseau."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

BOS, EOS, VOCAB_SIZE = 256, 257, 258
TOKENIZER = "utf8-bytes-v1"
MAX_CORPUS_BYTES = 16 * 1024 * 1024


def encode(text: str, *, end: bool = True) -> list[int]:
    return [BOS, *text.encode("utf-8"), *([EOS] if end else [])]


def decode(tokens: list[int]) -> str:
    return bytes(t for t in tokens if 0 <= t < 256).decode("utf-8", errors="replace")


def read_corpus(path: Path) -> tuple[dict, str]:
    """Les autorisations sont des déclarations à vérifier humainement, pas un audit de licence."""
    # Le laboratoire garde le corpus en RAM : lecture bornée pour le petit PC cible.
    with path.open("rb") as source:
        raw = source.read(MAX_CORPUS_BYTES + 1)
    if len(raw) > MAX_CORPUS_BYTES:
        raise ValueError("Corpus supérieur à 16 Mio : ce laboratoire attend un petit corpus local")
    groups: dict[str, list] = {"train": [], "validation": [], "test": []}
    ids, contents = set(), set()
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for field in ("id", "source", "license", "text"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Ligne {line_no} : {field} requis")
        if row.get("training_allowed") is not True:
            raise ValueError(f"Ligne {line_no} : autorisation explicite requise")
        if row.get("split") not in groups:
            raise ValueError(f"Ligne {line_no} : split train, validation ou test requis")
        normalized = " ".join(unicodedata.normalize("NFKC", row["text"]).casefold().split())
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        if row["id"] in ids or digest in contents:
            raise ValueError(f"Ligne {line_no} : identifiant ou texte dupliqué (fuite possible)")
        ids.add(row["id"])
        contents.add(digest)
        groups[row["split"]].append({
            "id": row["id"], "source": row["source"], "license": row["license"],
            "sha256": digest, "tokens": encode(row["text"]),
        })
    if not groups["train"] or not groups["validation"]:
        raise ValueError("Au moins un document distinct par split est requis")
    return groups, hashlib.sha256(raw).hexdigest()
