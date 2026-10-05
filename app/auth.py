"""Authentification du propriétaire : mot de passe -> jeton signé (HMAC-SHA256).

FICHIER PROTÉGÉ : l'évolution ne peut pas le modifier.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import time

from . import config


class AuthError(Exception):
    pass


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _key() -> bytes:
    s = config.settings
    secret = s.secret_key or s.owner_password
    if not secret:
        raise AuthError("OWNER_PASSWORD (ou SECRET_KEY) n'est pas configuré sur le serveur.")
    return hashlib.sha256(("kira-v2|" + secret).encode("utf-8")).digest()


def make_token(subject: str = "owner", days: int | None = None) -> str:
    days = days if days is not None else config.settings.session_days
    payload = {"sub": subject, "exp": int(time.time() + days * 86400)}
    body = _b64(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = _b64(hmac.new(_key(), body.encode("ascii"), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_token(token: str) -> dict | None:
    try:
        body, sig = token.split(".", 1)
        expected = _b64(hmac.new(_key(), body.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_unb64(body))
        if int(payload.get("exp", 0)) < time.time():
            return None
        return payload
    except (ValueError, AuthError, TypeError, json.JSONDecodeError):
        return None


def check_password(candidate: str) -> bool:
    expected = config.settings.owner_password
    if not expected or not candidate:
        return False
    a = hashlib.sha256(candidate.encode("utf-8")).digest()
    b = hashlib.sha256(expected.encode("utf-8")).digest()
    return hmac.compare_digest(a, b)


def check_cron_token(candidate: str) -> bool:
    expected = config.settings.cron_token
    if not expected or not candidate:
        return False
    return hmac.compare_digest(
        hashlib.sha256(candidate.encode("utf-8")).digest(),
        hashlib.sha256(expected.encode("utf-8")).digest(),
    )


class LoginThrottle:
    """Bloque une adresse après 5 échecs en 5 minutes."""

    def __init__(self, max_failures: int = 5, window: int = 300):
        self.max_failures = max_failures
        self.window = window
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, who: str) -> list[float]:
        cutoff = time.time() - self.window
        kept = [t for t in self._fails.get(who, []) if t >= cutoff]
        self._fails[who] = kept
        return kept

    def blocked(self, who: str) -> bool:
        with self._lock:
            return len(self._recent(who)) >= self.max_failures

    def fail(self, who: str) -> None:
        with self._lock:
            self._recent(who).append(time.time())

    def ok(self, who: str) -> None:
        with self._lock:
            self._fails.pop(who, None)


throttle = LoginThrottle()
