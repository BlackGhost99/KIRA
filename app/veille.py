"""Veille : KIRA lit des sources, résume, et range le tout dans une boîte « à valider ».

Rien n'entre dans ses connaissances sans la validation de Brice (sauf seuil
d'auto-validation, désactivé par défaut). Une fois validé, un élément devient
une connaissance (memories, kind='knowledge') que KIRA retrouve dans ses réponses.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from . import audit, config, memory, net
from .db import get_db, now_iso
from .llm import get_router
from .textutil import clip, extract_json
from .tools.web import html_to_text

MIN_CONTENT_CHARS = 120

# (adresse, nom, type, fiabilité, active)
DEFAULT_SOURCES = [
    ("https://rss.arxiv.org/rss/physics.ed-ph", "arXiv — enseignement de la physique", "rss", 0.9, 1),
    ("https://phys.org/rss-feed/physics-news/", "Phys.org — physique", "rss", 0.75, 1),
    ("https://api.quantamagazine.org/feed/", "Quanta Magazine — maths et physique", "rss", 0.9, 1),
    ("https://www.freecodecamp.org/news/rss/", "freeCodeCamp — programmation", "rss", 0.8, 1),
    ("https://github.blog/feed/", "GitHub Blog", "rss", 0.7, 1),
    ("https://hnrss.org/best", "Hacker News — les meilleurs", "rss", 0.6, 1),
    # Pages d'accueil de KIRA v1 : peu de contenu utile à lire telles quelles, gardées désactivées.
    ("https://ocw.mit.edu/", "MIT OpenCourseWare (page)", "page", 0.95, 0),
    ("https://www.khanacademy.org/", "Khan Academy (page)", "page", 0.9, 0),
    ("https://exo7.emath.fr/", "Exo7 — maths (page)", "page", 0.9, 0),
    ("https://cs50.harvard.edu/", "CS50 (page)", "page", 0.9, 0),
]


# -- sources ---------------------------------------------------------------
def seed_defaults() -> int:
    db = get_db()
    if db.q1("SELECT id FROM veille_sources LIMIT 1"):
        return 0
    for url, name, kind, score, active in DEFAULT_SOURCES:
        db.insert("veille_sources", {"url": url, "name": name, "kind": kind, "score": score, "active": active})
    return len(DEFAULT_SOURCES)


def list_sources() -> list[dict]:
    return get_db().q("SELECT * FROM veille_sources ORDER BY active DESC, id")


def add_source(url: str, name: str = "", kind: str = "rss", score: float = 0.5) -> dict:
    parts = urlparse((url or "").strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("L'adresse doit commencer par http:// ou https://")
    kind = kind if kind in ("rss", "page") else "rss"
    score = max(0.0, min(1.0, float(score)))
    row_id = get_db().insert(
        "veille_sources", {"url": url.strip(), "name": name.strip() or parts.netloc, "kind": kind, "score": score, "active": 1}
    )
    return get_db().q1("SELECT * FROM veille_sources WHERE id = ?", [row_id])


def update_source(source_id: int, **fields) -> dict | None:
    allowed = {k: v for k, v in fields.items() if k in ("name", "score", "active", "kind") and v is not None}
    if "active" in allowed:
        allowed["active"] = 1 if allowed["active"] else 0
    if "score" in allowed:
        allowed["score"] = max(0.0, min(1.0, float(allowed["score"])))
    get_db().update("veille_sources", source_id, allowed)
    return get_db().q1("SELECT * FROM veille_sources WHERE id = ?", [source_id])


def delete_source(source_id: int) -> bool:
    return get_db().delete("veille_sources", source_id) > 0


# -- lecture ---------------------------------------------------------------
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def strip_html(text: str) -> str:
    if not text:
        return ""
    return " ".join(BeautifulSoup(text, "html.parser").get_text(" ").split())


def parse_feed(body: bytes) -> list[dict]:
    """Lit un flux RSS 2.0, RDF/RSS 1.0 ou Atom."""
    root = ET.fromstring(body)
    entries: list[dict] = []
    for el in root.iter():
        if _local(el.tag) not in ("item", "entry"):
            continue
        entry = {"title": "", "link": "", "published": "", "summary": ""}
        for child in el:
            name = _local(child.tag)
            text = (child.text or "").strip()
            if name == "title":
                entry["title"] = text
            elif name == "link":
                href = child.get("href")
                if href:
                    if child.get("rel", "alternate") == "alternate" and not entry["link"]:
                        entry["link"] = href
                elif text:
                    entry["link"] = text
            elif name in ("pubDate", "published", "updated", "date"):
                entry["published"] = entry["published"] or text
            elif name in ("description", "summary", "content", "encoded"):
                if len(text) > len(entry["summary"]):
                    entry["summary"] = text
        entries.append(entry)
    return entries


def fetch_items(src: dict) -> list[dict]:
    page = net.safe_get(src["url"], max_bytes=2_000_000)
    if src["kind"] == "page":
        text = html_to_text(net.decode_body(page["body"], page["content_type"]), limit=6000)
        title = text.splitlines()[0] if text else src["name"]
        return [{"title": clip(title, 160), "url": src["url"], "published": "", "content": text}]
    return [
        {"title": e["title"], "url": e["link"] or src["url"], "published": e["published"], "content": strip_html(e["summary"])}
        for e in parse_feed(page["body"])
    ]


# -- résumé ----------------------------------------------------------------
_SYSTEM = (
    "Tu aides {owner} à faire sa veille d'apprentissage (centres d'intérêt : {interests}). "
    "Tu réponds uniquement par un objet JSON valide, sans texte autour."
)
_PROMPT = """Résume ce contenu pour {owner}.
Réponds en JSON : {{"resume": "2 à 4 phrases en français, factuelles, sans promesse ni jugement", \
"sujet": "physique|mathématiques|informatique|langues|droit|autre", \
"niveau": "débutant|intermédiaire|avancé", \
"pertinence": un nombre entre 0 et 1 (intérêt pour {owner} qui veut combler ses lacunes en sciences)}}

Titre : {title}
Contenu :
{content}"""


def summarize(item: dict, use_llm: bool = True) -> tuple[dict, str]:
    """Renvoie (résumé structuré, auteur du résumé : 'llm' ou 'extrait')."""
    s = config.settings
    fallback = {"resume": clip(item["content"], 350), "sujet": "autre", "niveau": "", "pertinence": 0.5}
    if not use_llm:
        return fallback, "extrait"
    try:
        res = get_router().complete(
            [_SYSTEM.format(owner=s.owner_name, interests=s.interests)],
            [{"role": "user", "content": _PROMPT.format(owner=s.owner_name, title=item["title"], content=clip(item["content"], 6000))}],
            None,
            tier="fast",
            max_tokens=500,
            purpose="veille",
        )
    except Exception as exc:  # noqa: BLE001 — pas d'IA disponible : on garde l'extrait
        audit.log("veille_llm_unavailable", {"error": str(exc)[:200]})
        return fallback, "extrait"
    data = extract_json(res.text)
    if not isinstance(data, dict) or not data.get("resume"):
        return fallback, "extrait"
    try:
        relevance = max(0.0, min(1.0, float(data.get("pertinence", 0.5))))
    except (TypeError, ValueError):
        relevance = 0.5
    return {
        "resume": clip(str(data["resume"]), 700),
        "sujet": str(data.get("sujet", "autre"))[:40],
        "niveau": str(data.get("niveau", ""))[:20],
        "pertinence": relevance,
    }, "llm"


# -- passage de veille -----------------------------------------------------
def run(max_per_source: int | None = None, max_total: int | None = None) -> dict:
    s = config.settings
    per_source = max_per_source or s.veille_max_per_source
    total_cap = max_total or s.veille_max_total
    db = get_db()
    stats = {"sources": 0, "new": 0, "skipped": 0, "errors": [], "by_llm": 0, "auto_validated": 0}
    use_llm = True
    for src in db.q("SELECT * FROM veille_sources WHERE active = 1 ORDER BY score DESC, id"):
        if stats["new"] >= total_cap:
            break
        stats["sources"] += 1
        try:
            items = fetch_items(src)
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"[:200]
            db.update("veille_sources", src["id"], {"last_run": now_iso(), "last_error": message})
            stats["errors"].append(f"{src['name']}: {message}")
            continue
        for item in items[:per_source]:
            if stats["new"] >= total_cap:
                break
            if len(item["content"]) < MIN_CONTENT_CHARS or not item["title"]:
                continue
            digest = hashlib.sha256(f"{item['url']}|{item['title']}|{item['content'][:500]}".encode()).hexdigest()
            if db.q1("SELECT id FROM veille_items WHERE content_hash = ?", [digest]):
                stats["skipped"] += 1
                continue
            summary, by = summarize(item, use_llm)
            if by == "extrait":
                use_llm = False  # inutile de réessayer l'IA pour chaque élément
            else:
                stats["by_llm"] += 1
            db.insert(
                "veille_items",
                {
                    "source_id": src["id"],
                    "title": clip(item["title"], 200),
                    "url": item["url"],
                    "published": item["published"],
                    "summary": summary["resume"],
                    "excerpt": clip(item["content"], 500),
                    "topic": summary["sujet"],
                    "level": summary["niveau"],
                    "score": round(float(src["score"]) * summary["pertinence"], 2),
                    "status": "pending",
                    "summarized_by": by,
                    "content_hash": digest,
                    "created_at": now_iso(),
                },
            )
            stats["new"] += 1
        db.update("veille_sources", src["id"], {"last_run": now_iso(), "last_error": None})
    if s.veille_auto_validate_min_score > 0:
        stats["auto_validated"] = validate_many(s.veille_auto_validate_min_score)
    db.kv_set("veille_last_run", now_iso())
    audit.log("veille_run", {k: v for k, v in stats.items()})
    return stats


# -- boîte « à valider » ---------------------------------------------------
def list_items(status: str = "pending", limit: int = 60) -> list[dict]:
    return get_db().q(
        "SELECT i.*, s.name AS source_name FROM veille_items i LEFT JOIN veille_sources s ON s.id = i.source_id "
        "WHERE i.status = ? ORDER BY i.score DESC, i.id DESC LIMIT ?",
        [status, limit],
    )


def counts() -> dict[str, int]:
    rows = get_db().q("SELECT status, COUNT(*) AS n FROM veille_items GROUP BY status")
    return {r["status"]: int(r["n"]) for r in rows}


def validate(item_id: int) -> dict | None:
    db = get_db()
    item = db.q1("SELECT * FROM veille_items WHERE id = ?", [item_id])
    if not item or item["status"] == "validated":
        return item
    db.update("veille_items", item_id, {"status": "validated"})
    memory.add("knowledge", f"{item['title']}\n{item['summary']}", tags=item.get("topic") or "", source=item["url"] or "")
    audit.log("veille_validated", {"item": item_id, "title": item["title"]}, actor="owner")
    return db.q1("SELECT * FROM veille_items WHERE id = ?", [item_id])


def reject(item_id: int) -> dict | None:
    db = get_db()
    db.update("veille_items", item_id, {"status": "rejected"})
    audit.log("veille_rejected", {"item": item_id}, actor="owner")
    return db.q1("SELECT * FROM veille_items WHERE id = ?", [item_id])


def validate_many(min_score: float) -> int:
    rows = get_db().q("SELECT id FROM veille_items WHERE status = 'pending' AND score >= ?", [min_score])
    for r in rows:
        validate(r["id"])
    return len(rows)
