"""Authentification du propriétaire : sessions opaques révocables et compatibilité HMAC.

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

# Sessions opaques : aucun jeton brut n'est conservé dans la base.
ACCESS_COOKIE = "kira_access"
REFRESH_COOKIE = "kira_refresh"


def _token_hash(token: str) -> str:
    # Un changement du mot de passe invalide aussi les sessions, même avec SECRET_KEY stable.
    key = hashlib.sha256(_key() + config.settings.owner_password.encode()).digest()
    return hmac.new(key, token.encode(), hashlib.sha256).hexdigest()


def _new_tokens() -> tuple[str, str]:
    import secrets
    return "kacc_" + secrets.token_urlsafe(32), "kref_" + secrets.token_urlsafe(48)


def create_session(user_agent: str = "") -> tuple[dict, str, str]:
    import uuid
    from .db import get_db, now_iso
    db = get_db()
    now = time.time()
    access, refresh = _new_tokens()
    session = {"id": str(uuid.uuid4()), "access_hash": _token_hash(access), "refresh_hash": _token_hash(refresh),
               "created_at": now_iso(), "access_expires_at": now + max(1, config.settings.access_minutes) * 60,
               "expires_at": now + max(1, config.settings.session_days) * 86400,
               "reauth_at": now, "user_agent": user_agent[:250]}
    with db.transaction():
        db.run("DELETE FROM owner_sessions WHERE expires_at <= ?", [now])
        db.insert("owner_sessions", session)
    return session, access, refresh


def authenticate_access(token: str) -> dict | None:
    from .db import get_db
    if not token or not token.startswith("kacc_") or len(token) > 200:
        return None
    now = time.time()
    return get_db().q1("SELECT * FROM owner_sessions WHERE access_hash = ? AND revoked_at IS NULL "
                       "AND access_expires_at > ? AND expires_at > ?", [_token_hash(token), now, now])


def refresh_session(token: str) -> tuple[dict, str, str] | None:
    from .db import get_db, now_iso
    if not token or not token.startswith("kref_") or len(token) > 200:
        return None
    db = get_db()
    hashed = _token_hash(token)
    now = time.time()
    with db.transaction():
        lock = " FOR UPDATE" if db.kind == "postgres" else ""
        row = db.q1("SELECT * FROM owner_sessions WHERE refresh_hash = ?" + lock, [hashed])
        if not row:
            used = db.q1("SELECT session_id FROM owner_refresh_used WHERE token_hash = ?", [hashed])
            if used:
                from . import audit
                audit.log("session_refresh_replay", {"id": used["session_id"]}, actor="owner")
                db.run("UPDATE owner_sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
                       [now_iso(), used["session_id"]])
            return None  # transaction commits the replay revocation
        if row["revoked_at"] or row["expires_at"] <= now:
            return None
        access, refresh = _new_tokens()
        db.run("INSERT INTO owner_refresh_used (token_hash, session_id, consumed_at) VALUES (?, ?, ?)",
               [hashed, row["id"], now])
        updates = {"access_hash": _token_hash(access), "refresh_hash": _token_hash(refresh),
                   "access_expires_at": min(row["expires_at"], now + max(1, config.settings.access_minutes) * 60)}
        db.update("owner_sessions", row["id"], updates)
        return {**row, **updates}, access, refresh


def revoke_session(session_id: str) -> None:
    from .db import get_db, now_iso
    get_db().run("UPDATE owner_sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL", [now_iso(), session_id])


def revoke_refresh(token: str) -> None:
    from .db import get_db
    if not token or len(token) > 200:
        return
    hashed = _token_hash(token)
    row = get_db().q1("SELECT id FROM owner_sessions WHERE refresh_hash = ?", [hashed])
    if not row:
        row = get_db().q1("SELECT session_id AS id FROM owner_refresh_used WHERE token_hash = ?", [hashed])
    if row:
        revoke_session(row["id"])


def list_sessions(current: str = "") -> list[dict]:
    from .db import get_db
    rows = get_db().q("SELECT id, created_at, expires_at, user_agent FROM owner_sessions "
                      "WHERE revoked_at IS NULL AND expires_at > ? ORDER BY created_at DESC", [time.time()])
    return [{**r, "current": r["id"] == current} for r in rows]


def confirm_session(session_id: str) -> None:
    from .db import get_db
    get_db().run("UPDATE owner_sessions SET reauth_at = ? WHERE id = ? AND revoked_at IS NULL", [time.time(), session_id])


def recent_confirmation(session: dict | None) -> bool:
    return bool(session and time.time() - session["reauth_at"] < max(1, config.settings.reauth_seconds))
