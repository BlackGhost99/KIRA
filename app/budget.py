"""Plafond quotidien de jetons : protège contre une facture imprévue.

FICHIER PROTÉGÉ : l'évolution ne peut pas le modifier.
"""
from __future__ import annotations

from . import config
from .db import get_db, now_iso, today


def used_today(paid_only: bool = False) -> int:
    row = get_db().q1(
        "SELECT COALESCE(SUM(input_tokens + output_tokens), 0) AS n FROM llm_usage WHERE day = ?" + (" AND cost_class = 'paid'" if paid_only else ""), [today()]
    )
    return int(row["n"]) if row else 0


def check() -> None:
    from .llm.base import BudgetExceeded  # import tardif : évite un import circulaire

    limit = config.settings.daily_token_budget
    if limit > 0 and used_today(paid_only=True) >= limit:
        raise BudgetExceeded(
            f"Plafond quotidien atteint ({limit} jetons). Il se réinitialise à minuit UTC ; "
            "vous pouvez le relever avec DAILY_TOKEN_BUDGET."
        )


def record(provider: str, model: str, input_tokens: int, output_tokens: int, purpose: str = "chat", cost_class: str = "paid") -> None:
    get_db().insert(
        "llm_usage",
        {
            "ts": now_iso(),
            "day": today(),
            "provider": provider,
            "model": model,
            "input_tokens": max(0, int(input_tokens or 0)),
            "output_tokens": max(0, int(output_tokens or 0)),
            "purpose": purpose,
            "cost_class": cost_class,
        },
    )


def summary() -> dict:
    rows = get_db().q(
        "SELECT provider, SUM(input_tokens) AS i, SUM(output_tokens) AS o FROM llm_usage "
        "WHERE day = ? GROUP BY provider",
        [today()],
    )
    return {
        "used": used_today(),
        "paid_used": used_today(paid_only=True),
        "limit": config.settings.daily_token_budget,
        "by_provider": {r["provider"]: int(r["i"] or 0) + int(r["o"] or 0) for r in rows},
    }
