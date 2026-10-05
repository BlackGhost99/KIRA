"""Mémoire de KIRA : profil de Brice, faits, connaissances validées, notes.

Recherche par mots (français) : texte intégral PostgreSQL en ligne, notation
maison en local. Tout est visible, modifiable et supprimable depuis l'application.
"""
from __future__ import annotations

import re
import unicodedata

from .db import get_db, now_iso

KINDS = ("profile", "fact", "knowledge", "note")

_STOP = set(
    """le la les un une des du de d l et ou mais donc or ni car que qui quoi dont où au aux en dans sur sous par pour
    avec sans ce cet cette ces mon ma mes ton ta tes son sa ses notre votre leur leurs je tu il elle on nous vous ils
    elles me te se lui y est es suis sont été etre être fait faire peux peut veux veut comment pourquoi quand combien
    quel quelle quels quelles the a an and or of to in is are was for with on that this it as be at by from""".split()
)


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def tokens(text: str, limit: int = 12) -> list[str]:
    seen: list[str] = []
    for w in re.findall(r"\w{3,}", _fold(text or "")):
        if w not in _STOP and w not in seen:
            seen.append(w)
        if len(seen) >= limit:
            break
    return seen


def _score(query_tokens: list[str], text: str) -> float:
    if not query_tokens:
        return 0.0
    words = re.findall(r"\w{3,}", _fold(text))
    if not words:
        return 0.0
    hits = 0.0
    for q in query_tokens:
        stem = q[:5] if len(q) > 5 else q
        if any(w == q for w in words):
            hits += 1.0
        elif any(w.startswith(stem) for w in words):
            hits += 0.6
    return hits / len(query_tokens)


def add(kind: str, content: str, tags: str = "", source: str = "") -> dict:
    kind = kind if kind in KINDS else "fact"
    content = (content or "").strip()
    if not content:
        raise ValueError("Le contenu est vide.")
    db = get_db()
    existing = db.q1("SELECT id FROM memories WHERE kind = ? AND content = ?", [kind, content])
    if existing:
        return get(existing["id"])
    ts = now_iso()
    new_id = db.insert(
        "memories",
        {"kind": kind, "content": content, "tags": tags or "", "source": source or "", "created_at": ts, "updated_at": ts},
    )
    return get(new_id)


def get(memory_id: int) -> dict | None:
    return get_db().q1("SELECT * FROM memories WHERE id = ?", [memory_id])


def update(memory_id: int, content: str | None = None, kind: str | None = None, tags: str | None = None) -> dict | None:
    values: dict = {"updated_at": now_iso()}
    if content is not None:
        if not content.strip():
            raise ValueError("Le contenu est vide.")
        values["content"] = content.strip()
    if kind in KINDS:
        values["kind"] = kind
    if tags is not None:
        values["tags"] = tags
    get_db().update("memories", memory_id, values)
    return get(memory_id)


def delete(memory_id: int) -> bool:
    return get_db().delete("memories", memory_id) > 0


def list_items(kind: str = "", limit: int = 100, offset: int = 0) -> list[dict]:
    if kind:
        return get_db().q(
            "SELECT * FROM memories WHERE kind = ? ORDER BY id DESC LIMIT ? OFFSET ?", [kind, limit, offset]
        )
    return get_db().q("SELECT * FROM memories ORDER BY id DESC LIMIT ? OFFSET ?", [limit, offset])


def counts() -> dict[str, int]:
    rows = get_db().q("SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind")
    return {r["kind"]: int(r["n"]) for r in rows}


def profile(limit: int = 40) -> list[dict]:
    return get_db().q("SELECT * FROM memories WHERE kind = 'profile' ORDER BY id DESC LIMIT ?", [limit])


def search(query: str, kinds: tuple[str, ...] | None = None, limit: int = 8) -> list[dict]:
    """Souvenirs les plus proches de ``query`` (les plus récents si la requête est vide)."""
    db = get_db()
    qt = tokens(query)
    kinds = kinds or KINDS
    if not qt:
        marks = ", ".join("?" for _ in kinds)
        return db.q(f"SELECT * FROM memories WHERE kind IN ({marks}) ORDER BY id DESC LIMIT ?", [*kinds, limit])
    if db.kind == "postgres":
        marks = ", ".join("?" for _ in kinds)
        tsq = " | ".join(re.sub(r"\W", "", t) for t in qt if re.sub(r"\W", "", t))
        rows = db.q(
            f"SELECT *, ts_rank(to_tsvector('french', content), to_tsquery('french', ?)) AS rank "
            f"FROM memories WHERE kind IN ({marks}) AND to_tsvector('french', content) @@ to_tsquery('french', ?) "
            f"ORDER BY rank DESC, id DESC LIMIT ?",
            [tsq, *kinds, tsq, limit],
        )
        for r in rows:
            r.pop("rank", None)
        return rows
    marks = ", ".join("?" for _ in kinds)
    rows = db.q(f"SELECT * FROM memories WHERE kind IN ({marks}) ORDER BY id DESC LIMIT 5000", list(kinds))
    scored = [(_score(qt, r["content"] + " " + r["tags"]), r) for r in rows]
    scored = [(s, r) for s, r in scored if s > 0]
    scored.sort(key=lambda sr: (-sr[0], -sr[1]["id"]))
    return [r for _, r in scored[:limit]]


def search_messages(query: str, limit: int = 6) -> list[dict]:
    """Retrouve d'anciens échanges (pour l'outil `recall`)."""
    db = get_db()
    qt = tokens(query)
    if not qt:
        return []
    if db.kind == "postgres":
        tsq = " | ".join(re.sub(r"\W", "", t) for t in qt if re.sub(r"\W", "", t))
        return db.q(
            "SELECT m.id, m.conversation_id, m.role, m.content, m.created_at FROM messages m "
            "WHERE to_tsvector('french', m.content) @@ to_tsquery('french', ?) ORDER BY m.id DESC LIMIT ?",
            [tsq, limit],
        )
    rows = db.q(
        "SELECT id, conversation_id, role, content, created_at FROM messages ORDER BY id DESC LIMIT 3000"
    )
    scored = [(_score(qt, r["content"]), r) for r in rows]
    scored = [(s, r) for s, r in scored if s > 0]
    scored.sort(key=lambda sr: (-sr[0], -sr[1]["id"]))
    return [r for _, r in scored[:limit]]
