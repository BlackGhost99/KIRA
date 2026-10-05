"""Fichiers produits par KIRA (graphiques, par exemple), stockés dans la base."""
from __future__ import annotations

import re
import secrets

from .db import get_db, now_iso

_ID = re.compile(r"^[0-9a-f]{32}$")


def save(name: str, mime: str, data: bytes) -> str:
    file_id = secrets.token_hex(16)  # 128 bits : l'adresse ne se devine pas
    get_db().run(
        "INSERT INTO files (id, mime, name, data, created_at) VALUES (?, ?, ?, ?, ?)",
        [file_id, mime, name, data, now_iso()],
    )
    return file_id


def get(file_id: str) -> dict | None:
    if not _ID.match(file_id or ""):
        return None
    return get_db().q1("SELECT id, mime, name, data FROM files WHERE id = ?", [file_id])


def url(file_id: str) -> str:
    return f"/api/files/{file_id}"
