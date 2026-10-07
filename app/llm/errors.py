"""Erreurs publiques bornées : ne jamais journaliser le corps d'une réponse fournisseur."""
import time
import math
from email.utils import parsedate_to_datetime

from .base import LLMError


def http_error(response, provider: str) -> LLMError:
    status = response.status_code
    category = "billing" if status == 402 else "rate_limit" if status == 429 else "configuration" if status in (401, 403, 404) else "http"
    try:
        error = response.json().get("error", {})
        code = str(error.get("code", "") or error.get("type", "")) if isinstance(error, dict) else ""
        if code in ("insufficient_quota", "billing_error", "credit_balance_too_low") or status == 402:
            category = "billing"
    except (ValueError, AttributeError, TypeError):
        pass
    delay = None
    header = response.headers.get("retry-after", "")
    try:
        delay = float(header)
        if not math.isfinite(delay):
            raise ValueError("nonfinite")
        delay = max(0, delay)
    except (ValueError, TypeError):
        delay = None
        try:
            delay = max(0, parsedate_to_datetime(header).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            pass
    return LLMError(f"{provider} : HTTP {status} ({category})", status, provider, delay, category)
