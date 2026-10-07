"""Routage cloud, coupe-circuits et budget réservé aux fournisseurs payants."""
from __future__ import annotations

import threading
import time

import requests

from .. import audit, budget, config
from ..db import get_db
from .anthropic_provider import AnthropicProvider
from .base import BudgetExceeded, LLMError, LLMResult, LLMUnavailable, Provider, ToolSpec
from .openai_compat import (deepseek_provider, gemini_provider, groq_provider,
                            ollama_provider, openai_provider, openrouter_provider)

MODES = ("QUALITY", "ECONOMY", "PRIVATE")


class Router:
    def __init__(self, providers: list[Provider] | None = None):
        self.providers = providers if providers is not None else [
            AnthropicProvider(), openai_provider(), deepseek_provider(),
            gemini_provider(), groq_provider(), openrouter_provider(), ollama_provider(),
        ]
        self._cool: dict[str, float] = {}
        self._disabled: dict[str, str] = {}
        self._reason: dict[str, str] = {}
        self._metrics: dict[str, dict] = {}
        self._lock = threading.RLock()

    def mode(self) -> str:
        value = get_db().kv_get("llm_routing_mode", config.settings.llm_routing_mode).upper()
        return value if value in MODES else "QUALITY"

    def trusted(self, p: Provider) -> bool:
        return p.name in {n.strip() for n in config.settings.llm_trusted_providers.split(",") if n.strip()}

    def _ordered(self) -> list[Provider]:
        rank = {name.strip(): i for i, name in enumerate(config.settings.llm_priority.split(","))}
        return sorted(self.providers, key=lambda p: rank.get(p.name, len(rank)))

    def active(self) -> list[Provider]:
        return [p for p in self._ordered() if p.configured()]

    def _ready(self, p: Provider) -> bool:
        with self._lock:
            if p.name in self._disabled and self._disabled[p.name] != p.fingerprint():
                self._disabled.pop(p.name, None)
                self._cool.pop(p.name, None)
                self._reason.pop(p.name, None)
            return p.name not in self._disabled and self._cool.get(p.name, 0) <= time.time()

    def status(self) -> list[dict]:
        with self._lock:
            return [{"name": p.name, "configured": p.configured(),
                     "model": p.model_for("default") if p.configured() else "",
                     "ready": self._ready(p), "cost_class": p.cost_class(),
                     "vision": p.vision, "tools": p.supports_tools, "trusted": self.trusted(p),
                     "cooldown_s": max(0, int(self._cool.get(p.name, 0) - time.time())),
                     "disabled": p.name in self._disabled, "reason": self._reason.get(p.name, ""),
                     **self._metrics.get(p.name, {})} for p in self._ordered()]

    def _trip(self, p: Provider, exc: Exception) -> str:
        status = getattr(exc, "status", None)
        category = getattr(exc, "category", "") or (
            "configuration" if status in (401, 403, 404) else "rate_limit" if status == 429 else "network" if status is None else "http")
        delay = getattr(exc, "retry_after", None)
        with self._lock:
            if category == "configuration":
                self._disabled[p.name] = p.fingerprint()
            # Respect Retry-After, including provider delays longer than our default.
            self._cool[p.name] = time.time() + max(1, delay if delay is not None else 600 if category == "billing" else 120)
            self._reason[p.name] = category
            metrics = self._metrics.setdefault(p.name, {})
            metrics["failures"] = metrics.get("failures", 0) + 1
        return f"{p.name}: {category}" + (f" (HTTP {status})" if status else "")

    def complete(self, system: list[str], messages: list[dict], tools: list[ToolSpec] | None = None,
                 tier: str = "default", max_tokens: int = 4096, purpose: str = "chat",
                 private: bool = False) -> LLMResult:
        active = self.active()
        if not active:
            raise LLMUnavailable("Aucun fournisseur configuré. Ajoutez ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY ou OPENROUTER_API_KEY.")
        mode = "PRIVATE" if private else self.mode()
        vision = any(m.get("images") for m in messages)
        candidates = [p for p in active if (not vision or p.vision) and (not tools or p.supports_tools)
                      and (mode != "PRIVATE" or self.trusted(p))]
        if mode == "ECONOMY":
            if not config.settings.llm_economy_allow_paid:
                candidates = [p for p in candidates if p.cost_class() != "paid"]
            candidates.sort(key=lambda p: p.cost_class() == "paid")
        errors: list[str] = []
        budget_blocked = False
        queue = [p for p in candidates if self._ready(p)]
        while queue:
            p = queue.pop(0)
            if p.cost_class() == "paid":
                try:
                    budget.check()
                except BudgetExceeded:
                    budget_blocked = True
                    continue
            started = time.monotonic()
            try:
                result = p.complete(system, messages, tools, tier, max_tokens)
            except (LLMError, requests.RequestException) as exc:
                public_error = self._trip(p, exc)
                errors.append(public_error)
                audit.log("llm_error", {"provider": p.name, "error": public_error, "purpose": purpose})
                if p.cost_class() == "paid":
                    queue.sort(key=lambda candidate: candidate.cost_class() == "paid")
                continue
            with self._lock:
                self._cool.pop(p.name, None)
                self._reason.pop(p.name, None)
                metrics = self._metrics.setdefault(p.name, {})
                metrics["successes"] = metrics.get("successes", 0) + 1
                metrics["latency_ms"] = round((time.monotonic() - started) * 1000)
            budget.record(result.provider, result.model, result.input_tokens, result.output_tokens, purpose, p.cost_class())
            if errors or budget_blocked:
                audit.log("provider_failover", {"used": p.name, "failed": errors, "budget_blocked": budget_blocked, "purpose": purpose})
            return result
        if budget_blocked and not errors:
            raise BudgetExceeded("Budget payant atteint ; aucun fournisseur gratuit compatible n'est disponible.")
        raise LLMUnavailable("Aucun fournisseur compatible disponible (mode " + mode + "). " + " | ".join(errors), errors)


_router: Router | None = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router()
    return _router


def set_router(router: Router | None) -> None:
    global _router
    _router = router
