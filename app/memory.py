"""Mémoire de KIRA : profil de Brice, faits, connaissances validées, notes.

Recherche hybride : texte intégral et pgvector dans Supabase ; SQLite sert aux
tests. Les versions sont conservées ; archives et souvenirs remplacés ne sont
pas injectés dans la conversation. L'effacement explicite reste définitif.
"""
from __future__ import annotations

import re
import unicodedata
import uuid

from . import audit, config, embeddings
from .db import get_db, jdump, jload, now_iso

KINDS = ("profile", "fact", "knowledge", "note", "identity", "episodic", "procedural", "project", "goal", "experience", "relationship")

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


def _weight(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("Importance et confiance doivent être entre 0 et 1.") from None
    if not 0 <= number <= 1:
        raise ValueError("Importance et confiance doivent être entre 0 et 1.")
    return number


def _record(item: dict):
    get_db().insert("memory_history", {"memory_id": item["id"], "version": item["version"],
                    "snapshot": jdump(item), "created_at": now_iso()})


def _locked(memory_id: int) -> dict | None:
    db = get_db()
    return db.q1("SELECT * FROM memories WHERE id = ?" + (" FOR UPDATE" if db.kind == "postgres" else ""), [memory_id])


def add(kind: str, content: str, tags: str = "", source: str = "", *, importance: float = 0.5,
        confidence: float = 1.0) -> dict:
    kind = kind if kind in KINDS else "fact"
    content = (content or "").strip()
    if not content:
        raise ValueError("Le contenu est vide.")
    db = get_db()
    importance, confidence = _weight(importance), _weight(confidence)
    with db.transaction():
        existing = db.q1("SELECT * FROM memories WHERE kind = ? AND content = ? AND status = 'active'", [kind, content])
        if existing:
            return existing
        ts = now_iso()
        new_id = db.insert("memories", {"kind": kind, "content": content, "tags": tags or "", "source": source or "",
                           "importance": importance, "confidence": confidence, "created_at": ts, "updated_at": ts})
        item = get(new_id)
        _record(item)
    index_item(new_id)
    audit.log("memory_created", {"id": new_id, "kind": kind})
    return item


def get(memory_id: int) -> dict | None:
    return get_db().q1("SELECT * FROM memories WHERE id = ?", [memory_id])


def update(memory_id: int, content: str | None = None, kind: str | None = None, tags: str | None = None,
           *, importance: float | None = None, confidence: float | None = None, status: str | None = None) -> dict | None:
    values: dict = {"updated_at": now_iso()}
    if content is not None:
        if not content.strip():
            raise ValueError("Le contenu est vide.")
        values["content"] = content.strip()
    if kind in KINDS:
        values["kind"] = kind
    if tags is not None:
        values["tags"] = tags
    for name, value in (("importance", importance), ("confidence", confidence)):
        if value is not None:
            values[name] = _weight(value)
    if status is not None:
        if status not in ("active", "archived"):
            raise ValueError("État de souvenir invalide.")
        values["status"] = status
        if status == "active":
            values["superseded_by"] = None
    db = get_db()
    with db.transaction():
        old = _locked(memory_id)
        if not old:
            return None
        if not db.q1("SELECT id FROM memory_history WHERE memory_id = ? AND version = ?", [memory_id, old["version"]]):
            _record(old)  # conserve aussi la version d'un souvenir créé avant la V3
        values["version"] = old["version"] + 1
        db.update("memories", memory_id, values)
        item = get(memory_id)
        _record(item)
        if content is not None:
            db.run("DELETE FROM memory_embeddings WHERE memory_id = ?", [memory_id])
    if content is not None:
        index_item(memory_id)
    audit.log("memory_revised", {"id": memory_id, "version": item["version"]})
    return item


def history(memory_id: int) -> list[dict]:
    rows = get_db().q("SELECT snapshot FROM memory_history WHERE memory_id = ? ORDER BY version", [memory_id])
    versions = [jload(r["snapshot"], {}) for r in rows]
    current = get(memory_id)
    if current and not any(v.get("version") == current["version"] for v in versions):
        versions.append(current)
    return versions


def supersede(memory_id: int, content: str, *, kind: str | None = None, source: str = "manuel") -> dict:
    """Remplacement explicite : conserve l'ancienne information et relie la nouvelle."""
    content = (content or "").strip()
    if not content:
        raise ValueError("Le contenu est vide.")
    if kind is not None and kind not in KINDS:
        raise ValueError("Type de souvenir invalide.")
    db = get_db()
    with db.transaction():
        old = _locked(memory_id)
        if not old or old["status"] != "active":
            raise ValueError("Le souvenir à remplacer doit être actif.")
        if old["content"] == content and (kind is None or kind == old["kind"]):
            return old
        if not db.q1("SELECT id FROM memory_history WHERE memory_id = ? AND version = ?", [memory_id, old["version"]]):
            _record(old)
        ts = now_iso()
        new_id = db.insert("memories", {"kind": kind or old["kind"], "content": content, "tags": old["tags"], "source": source,
                           "importance": old["importance"], "confidence": old["confidence"],
                           "created_at": ts, "updated_at": ts})
        item = get(new_id)
        _record(item)
        db.update("memories", memory_id, {"status": "superseded", "superseded_by": new_id,
                  "version": old["version"] + 1, "updated_at": ts})
        _record(get(memory_id))
    index_item(new_id)
    audit.log("memory_superseded", {"id": memory_id, "replacement": new_id})
    return item


def delete(memory_id: int) -> bool:
    deleted = get_db().delete("memories", memory_id) > 0
    if deleted:
        audit.log("memory_erased", {"id": memory_id})
    return deleted


def list_items(kind: str = "", limit: int = 100, offset: int = 0, status: str = "active") -> list[dict]:
    where, params = [], []
    if kind:
        where.append("kind = ?")
        params.append(kind)
    if status:
        where.append("status = ?")
        params.append(status)
    clause = " WHERE " + " AND ".join(where) if where else ""
    return get_db().q("SELECT * FROM memories" + clause + " ORDER BY id DESC LIMIT ? OFFSET ?", [*params, limit, offset])


def counts(status: str = "active") -> dict[str, int]:
    rows = get_db().q("SELECT kind, COUNT(*) AS n FROM memories" + (" WHERE status = ?" if status else "") +
                      " GROUP BY kind", [status] if status else [])
    return {r["kind"]: int(r["n"]) for r in rows}


def profile(limit: int = 40) -> list[dict]:
    return get_db().q("SELECT * FROM memories WHERE kind = 'profile' AND status = 'active' ORDER BY id DESC LIMIT ?", [limit])


def _lexical(query: str, kinds: tuple[str, ...], limit: int, status: str = "active") -> list[dict]:
    """Souvenirs les plus proches de ``query`` (les plus récents si la requête est vide)."""
    db = get_db()
    qt = tokens(query)
    condition = " AND status = ?" if status else ""
    status_params = [status] if status else []
    if not qt:
        marks = ", ".join("?" for _ in kinds)
        return db.q(f"SELECT * FROM memories WHERE kind IN ({marks}){condition} ORDER BY id DESC LIMIT ?", [*kinds, *status_params, limit])
    if db.kind == "postgres":
        marks = ", ".join("?" for _ in kinds)
        tsq = " | ".join(re.sub(r"\W", "", t) for t in qt if re.sub(r"\W", "", t))
        rows = db.q(
            f"SELECT *, ts_rank(to_tsvector('french', content), to_tsquery('french', ?)) AS rank "
            f"FROM memories WHERE kind IN ({marks}){condition} AND to_tsvector('french', content) @@ to_tsquery('french', ?) "
            f"ORDER BY rank DESC, id DESC LIMIT ?",
            [tsq, *kinds, *status_params, tsq, limit],
        )
        for r in rows:
            r.pop("rank", None)
        return rows
    marks = ", ".join("?" for _ in kinds)
    rows = db.q(f"SELECT * FROM memories WHERE kind IN ({marks}){condition} ORDER BY id DESC LIMIT 5000", [*kinds, *status_params])
    scored = [(_score(qt, r["content"] + " " + r["tags"]), r) for r in rows]
    scored = [(s, r) for s, r in scored if s > 0]
    scored.sort(key=lambda sr: (-sr[0], -sr[1]["id"]))
    return [r for _, r in scored[:limit]]


def index_item(memory_id: int) -> bool:
    """Calcule hors transaction ; refuse un résultat devenu périmé entre-temps."""
    if not embeddings.configured():
        return False
    db = get_db()
    if db.kind == "postgres" and not db.vector_schema:
        return False  # pas de transmission inutile tant que pgvector n'est pas activé
    item = get(memory_id)
    if not item or item["status"] != "active":
        return False
    try:
        vector = embeddings.embed(item["content"])
    except embeddings.EmbeddingUnavailable:
        audit.log("memory_embedding_unavailable", {"id": memory_id})
        return False
    with db.transaction():
        current = _locked(memory_id)
        if not current or current["content"] != item["content"] or current["status"] != "active":
            return False
        values = [memory_id, config.settings.embedding_model, len(vector), embeddings.content_hash(item["content"]), jdump(vector), now_iso()]
        db.run("INSERT INTO memory_embeddings (memory_id, model, dimensions, content_hash, vector_json, updated_at) "
               "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (memory_id) DO UPDATE SET model = excluded.model, "
               "dimensions = excluded.dimensions, content_hash = excluded.content_hash, "
               "vector_json = excluded.vector_json, updated_at = excluded.updated_at", values)
        if db.vector_schema:
            db.run(f"UPDATE memory_embeddings SET embedding = ?::{db.vector_schema}.vector WHERE memory_id = ?", [jdump(vector), memory_id])
    return True


def backfill(limit: int = 50) -> dict:
    """Petit lot reprenable des anciens souvenirs, sans recalcul des vecteurs valides."""
    db = get_db()
    rows = db.q("SELECT m.* FROM memories m LEFT JOIN memory_embeddings e ON e.memory_id = m.id "
                "WHERE m.status = 'active' AND (e.memory_id IS NULL OR e.model != ? OR e.dimensions != ?) "
                "ORDER BY m.id LIMIT ?", [config.settings.embedding_model, config.settings.embedding_dimensions, min(max(limit, 1), 100)])
    indexed = 0
    for row in rows:
        if not index_item(row["id"]):
            break  # panne : éviter une avalanche d'appels et reprendre au prochain lot
        indexed += 1
    return {"indexed": indexed, "remaining_in_batch": len(rows) - indexed}


def semantic_status() -> dict:
    db = get_db()
    return {"configured": embeddings.configured(), "pgvector": bool(db.vector_schema),
            "model": config.settings.embedding_model, "dimensions": config.settings.embedding_dimensions,
            "indexed": db.q1("SELECT COUNT(*) AS n FROM memory_embeddings e JOIN memories m ON m.id = e.memory_id "
                              "WHERE e.model = ? AND e.dimensions = ? AND m.status = 'active'",
                              [config.settings.embedding_model, config.settings.embedding_dimensions])["n"]}


def _semantic(query: str, kinds: tuple[str, ...], limit: int) -> list[dict]:
    db = get_db()
    if not embeddings.configured() or (db.kind == "postgres" and not db.vector_schema):
        return []
    s = config.settings
    # Évite tout appel à l'endpoint si aucun souvenir compatible n'est indexé.
    if not db.q1("SELECT memory_id FROM memory_embeddings WHERE model = ? AND dimensions = ? LIMIT 1", [s.embedding_model, s.embedding_dimensions]):
        return []
    try:
        vector = embeddings.embed(query)
    except embeddings.EmbeddingUnavailable:
        return []
    marks = ", ".join("?" for _ in kinds)
    where = f"m.kind IN ({marks}) AND m.status = 'active' AND e.model = ? AND e.dimensions = ?"
    params = [*kinds, s.embedding_model, len(vector)]
    if db.kind == "postgres":
        # Embeddings non typés en dimensions pour permettre de changer de modèle.
        op = f"OPERATOR({db.vector_schema}.<=>)"
        rows = db.q(f"SELECT m.*, 1 - (e.embedding {op} ?::{db.vector_schema}.vector) AS similarity "
                    f"FROM memories m JOIN memory_embeddings e ON e.memory_id = m.id WHERE {where} "
                    "AND e.embedding IS NOT NULL ORDER BY similarity DESC, m.id DESC LIMIT ?",
                    [jdump(vector), *params, limit])
    else:
        rows = db.q(f"SELECT m.*, e.vector_json FROM memories m JOIN memory_embeddings e ON e.memory_id = m.id WHERE {where}", params)
        for r in rows:
            stored = jload(r.pop("vector_json"), [])
            r["similarity"] = sum(a * b for a, b in zip(vector, stored)) if len(stored) == len(vector) else -1
        rows.sort(key=lambda r: (-r["similarity"], -r["id"]))
    matched = []
    for r in rows:
        similarity = r.pop("similarity")
        if similarity is not None and similarity >= s.memory_semantic_threshold:
            matched.append(r)
    return matched[:limit]


def search(query: str, kinds: tuple[str, ...] | None = None, limit: int = 8, *, status: str = "active") -> list[dict]:
    """Fusion RRF : mots + sens, puis importance/confiance/récence comme départage."""
    if limit <= 0:
        return []
    kinds = tuple(k for k in (kinds or KINDS) if k in KINDS)
    if not kinds:
        return []
    lexical = _lexical(query, kinds, max(limit * 3, 24), status)
    if not tokens(query) or status != "active":
        return lexical[:limit]
    semantic = _semantic(query, kinds, max(limit * 3, 24))
    scores, items = {}, {}
    for candidates in (lexical, semantic):
        for rank, row in enumerate(candidates, 1):
            mid = row["id"]
            items[mid] = row
            scores[mid] = scores.get(mid, 0) + 1 / (60 + rank)
    def priority(mid):
        row = items[mid]
        return (scores[mid], row["importance"] * row["confidence"], row["updated_at"], mid)
    return [items[mid] for mid in sorted(items, key=priority, reverse=True)[:limit]]


def mark_used(items: list[dict]):
    """La consultation de l'interface ne change pas les compteurs d'usage cognitif."""
    db = get_db()
    for mid in {r["id"] for r in items}:
        db.run("UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?", [now_iso(), mid])


def identity() -> dict:
    """Identifiant stable du cerveau cloud, partagé par tous les appareils."""
    db = get_db()
    initial = {"id": uuid.uuid4().hex, "name": "KIRA", "created_at": now_iso(), "version": 1}
    db.run("INSERT INTO kv (k, v) VALUES (?, ?) ON CONFLICT (k) DO NOTHING", ["kira_identity", jdump(initial)])
    return jload(db.kv_get("kira_identity"), initial)


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
