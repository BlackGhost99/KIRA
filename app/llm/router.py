"""Routeur d'IA : essaie les fournisseurs dans l'ordre et bascule en cas de panne.

Reprend l'idée du « coupe-circuit » de KIRA v1 : un fournisseur qui échoue est
mis de côté quelques minutes au lieu d'être rappelé en boucle.
"""
from __future__ import annotations

import time

import requests

from .. import audit, budget, config
from .anthropic_provider import AnthropicProvider
from .base import LLMError, LLMResult, LLMUnavailable, Provider, ToolSpec
from .openai_compat import deepseek_provider, groq_provider, ollama_provider, openai_provider


class Router:
    def __init__(self, providers: list[Provider] | None = None):
        self.providers = providers if providers is not None else [
            AnthropicProvider(),
            openai_provider(),
            deepseek_provider(),
            groq_provider(),
            ollama_provider(),
        ]
        self._cool: dict[str, float] = {}

    # -- état --------------------------------------------------------------
    def _ordered(self) -> list[Provider]:
        order = [n.strip() for n in config.settings.llm_priority.split(",") if n.strip()]
        rank = {name: i for i, name in enumerate(order)}
        return sorted(self.providers, key=lambda p: rank.get(p.name, len(rank)))

    def active(self) -> list[Provider]:
        return [p for p in self._ordered() if p.configured()]

    def status(self) -> list[dict]:
        now = time.time()
        return [
            {
                "name": p.name,
                "configured": p.configured(),
                "model": p.model_for("default") if p.configured() else "",
                "cooldown_s": max(0, int(self._cool.get(p.name, 0) - now)),
            }
            for p in self._ordered()
        ]

    def _trip(self, name: str, exc: Exception) -> None:
        status = getattr(exc, "status", None)
        # 429 / 5xx / réseau : pause courte ; erreur de configuration (401, 404...) : pause longue
        seconds = 120 if status is None or status == 429 or status >= 500 else 600
        self._cool[name] = time.time() + seconds

    # -- appel -------------------------------------------------------------
    def complete(self, system: list[str], messages: list[dict], tools: list[ToolSpec] | None = None,
                 tier: str = "default", max_tokens: int = 4096, purpose: str = "chat") -> LLMResult:
        budget.check()
        active = self.active()
        if not active:
            raise LLMUnavailable(
                "Aucun fournisseur d'IA n'est configuré. Ajoutez ANTHROPIC_API_KEY (ou OPENAI_API_KEY, "
                "DEEPSEEK_API_KEY, GROQ_API_KEY) dans les variables d'environnement."
            )
        errors: list[str] = []
        now = time.time()
        fresh = [p for p in active if self._cool.get(p.name, 0) <= now]
        resting = [p for p in active if p not in fresh]
        for p in fresh + resting:  # en dernier recours, on retente ceux mis de côté
            try:
                result = p.complete(system, messages, tools, tier, max_tokens)
            except (LLMError, requests.RequestException) as exc:
                self._trip(p.name, exc)
                errors.append(f"{p.name}: {exc}")
                audit.log("llm_error", {"provider": p.name, "error": str(exc)[:300], "purpose": purpose})
                continue
            self._cool.pop(p.name, None)
            budget.record(result.provider, result.model, result.input_tokens, result.output_tokens, purpose)
            if errors:
                audit.log("provider_failover", {"used": p.name, "failed": errors, "purpose": purpose})
            return result
        raise LLMUnavailable("Aucun fournisseur d'IA n'a pu répondre : " + " | ".join(errors), errors)


_router: Router | None = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router()
    return _router


def set_router(router: Router | None) -> None:
    global _router
    _router = router
