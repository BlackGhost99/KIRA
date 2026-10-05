"""Consolidation : chaque nuit, KIRA relit les échanges du jour et retient ce qui dure
(niveau, lacunes, objectifs, préférences de Brice). Tout est visible et supprimable dans l'onglet Mémoire.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from . import audit, config, memory
from .db import get_db
from .llm import get_router
from .textutil import clip, extract_json

_SYSTEM = (
    "Tu extrais la mémoire durable d'un assistant personnel. Tu réponds uniquement par un objet JSON valide."
)
_PROMPT = """Voici les échanges récents entre {owner} et son assistant KIRA, puis ce que KIRA sait déjà de lui.

Extrais au plus 8 éléments NOUVEAUX et durables : niveau dans une matière, lacunes constatées, objectifs, préférences \
d'apprentissage, projets en cours, faits personnels utiles. Ignore les détails passagers et ce qui est déjà connu.
Réponds en JSON : {{"items": [{{"kind": "profile" ou "fact", "content": "une phrase claire et autonome"}}]}}

Déjà connu :
{known}

Échanges :
{transcript}"""


def consolidate(hours: int = 24) -> dict:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")
    rows = get_db().q("SELECT role, content FROM messages WHERE created_at >= ? ORDER BY id", [since])
    if sum(1 for r in rows if r["role"] == "user") < 3:
        return {"added": 0, "skipped": "pas assez d'échanges récents"}
    owner = config.settings.owner_name
    transcript = "\n".join(
        f"{owner if r['role'] == 'user' else 'KIRA'} : {clip(r['content'], 600)}" for r in rows
    )[-14000:]
    known = "\n".join(f"- {clip(p['content'], 200)}" for p in memory.profile(30)) or "(rien)"
    res = get_router().complete(
        [_SYSTEM],
        [{"role": "user", "content": _PROMPT.format(owner=owner, known=known, transcript=transcript)}],
        None,
        tier="fast",
        max_tokens=800,
        purpose="consolidation",
    )
    data = extract_json(res.text)
    items = data.get("items", []) if isinstance(data, dict) else []
    added = 0
    for it in items[:8]:
        if isinstance(it, dict) and isinstance(it.get("content"), str) and it["content"].strip():
            before = memory.counts()
            memory.add(it.get("kind") if it.get("kind") in ("profile", "fact") else "fact", it["content"], source="consolidation")
            added += 1 if memory.counts() != before else 0
    audit.log("consolidation", {"added": added, "candidates": len(items)})
    return {"added": added, "candidates": len(items)}
