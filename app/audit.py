"""Journal d'audit : « aucune action sans trace »."""
from __future__ import annotations

import logging

from .db import get_db, jdump, jload, now_iso

_log = logging.getLogger("kira.audit")


def log(action: str, details: dict | None = None, actor: str = "kira") -> None:
    """Écrit une ligne de journal. Ne lève jamais d'exception."""
    try:
        get_db().insert(
            "audit",
            {"ts": now_iso(), "actor": actor, "action": action, "details": jdump(details or {})},
        )
    except Exception:  # noqa: BLE001
        _log.exception("écriture du journal impossible (%s)", action)


def recent(limit: int = 100, prefix: str = "") -> list[dict]:
    if prefix:
        rows = get_db().q(
            "SELECT id, ts, actor, action, details FROM audit WHERE action LIKE ? ORDER BY id DESC LIMIT ?",
            [prefix + "%", limit],
        )
    else:
        rows = get_db().q("SELECT id, ts, actor, action, details FROM audit ORDER BY id DESC LIMIT ?", [limit])
    for r in rows:
        r["details"] = jload(r["details"], {})
    return rows


def counts_since(days: int = 14) -> dict[str, int]:
    from datetime import datetime, timedelta, timezone

    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    rows = get_db().q("SELECT action, COUNT(*) AS n FROM audit WHERE ts >= ? GROUP BY action", [since])
    return {r["action"]: int(r["n"]) for r in rows}
