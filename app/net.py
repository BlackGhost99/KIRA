"""Accès réseau sortant, avec protection contre les adresses internes (SSRF).

FICHIER PROTÉGÉ : l'évolution ne peut pas le modifier.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests

USER_AGENT = "KIRA/2.0 (assistant personnel; lecture de pages publiques)"
ALLOWED_PORTS = {80, 443, 8080, 8443}

_session: requests.Session | None = None


class BlockedURL(Exception):
    pass


def session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = USER_AGENT
    return _session


def set_session(s) -> None:
    """Remplace la session HTTP (utilisé par les tests)."""
    global _session
    _session = s


def assert_public_url(url: str) -> None:
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise BlockedURL("URL non autorisée (seuls http et https sont permis).")
    port = parts.port or (443 if parts.scheme == "https" else 80)
    if port not in ALLOWED_PORTS:
        raise BlockedURL(f"Port {port} non autorisé.")
    try:
        infos = socket.getaddrinfo(parts.hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise BlockedURL(f"Hôte introuvable : {parts.hostname}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if getattr(ip, "ipv4_mapped", None):
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise BlockedURL("Adresse interne ou réservée : accès refusé.")


def decode_body(body: bytes, content_type: str = "") -> str:
    charset = "utf-8"
    lowered = (content_type or "").lower()
    if "charset=" in lowered:
        charset = lowered.split("charset=", 1)[1].split(";")[0].strip() or "utf-8"
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def safe_get(url: str, max_bytes: int = 1_500_000, timeout: int = 15, headers: dict | None = None,
             max_redirects: int = 4) -> dict:
    """GET public avec contrôle de chaque redirection et limite de taille."""
    current = url
    for _ in range(max_redirects + 1):
        assert_public_url(current)
        resp = session().get(current, headers=headers or {}, timeout=timeout, stream=True, allow_redirects=False)
        try:
            location = resp.headers.get("location")
            if resp.status_code in (301, 302, 303, 307, 308) and location:
                current = urljoin(current, location)
                continue
            resp.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            truncated = False
            for chunk in resp.iter_content(65536):
                total += len(chunk)
                if total > max_bytes:
                    chunks.append(chunk[: max(0, len(chunk) - (total - max_bytes))])
                    truncated = True
                    break
                chunks.append(chunk)
            return {
                "url": current,
                "content_type": resp.headers.get("content-type", ""),
                "body": b"".join(chunks),
                "truncated": truncated,
            }
        finally:
            resp.close()
    raise BlockedURL("Trop de redirections.")
