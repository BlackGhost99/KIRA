"""Accueil de KIRA : bilan du jour, ce qu'il a appris, et de quoi reprendre la conversation."""
from __future__ import annotations

from . import config, memory, veille
from .db import get_db
from .textutil import clip, local_now


def greeting(hour: int, name: str) -> str:
    if hour < 5:
        return f"Encore debout, {name} ?"
    if hour < 12:
        return f"Bonjour {name}"
    if hour < 18:
        return f"Bon après-midi {name}"
    return f"Bonsoir {name}"


def suggestions(profile: list[dict], last_title: str | None) -> list[str]:
    topics = [t.strip() for t in config.settings.interests.split(",") if t.strip()] or ["physique"]
    out: list[str] = []
    if last_title:
        out.append(f"On reprend : {clip(last_title, 70)}")
    if profile:
        out.append(f"Fais le point avec moi : {clip(profile[0]['content'], 90)}")
    out.append(f"Pose-moi trois questions pour repérer mes lacunes en {topics[0]}")
    out.append("Explique-moi un concept que je ne connais pas encore, avec un exemple concret")
    out.append(f"Qu'est-ce que tu as appris de nouveau en {topics[min(1, len(topics) - 1)]} ?")
    return out[:4]


def build() -> dict:
    s = config.settings
    now = local_now(s.timezone)
    prof = memory.profile(5)
    last = get_db().q1("SELECT title FROM conversations ORDER BY updated_at DESC LIMIT 1")
    pending = veille.list_items("pending", limit=3)
    counts = veille.counts()
    return {
        "greeting": greeting(now.hour, s.owner_name),
        "veille": {
            "pending": counts.get("pending", 0),
            "top": [
                {"id": i["id"], "title": i["title"], "summary": clip(i["summary"] or "", 220), "score": i["score"],
                 "url": i["url"], "topic": i["topic"]}
                for i in pending
            ],
            "last_run": get_db().kv_get("veille_last_run") or None,
        },
        "known": len(memory.profile(1000)),
        "suggestions": suggestions(prof, last["title"] if last else None),
    }
