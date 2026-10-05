"""Point d'entrée : application ASGI (API + interface web installable).

Lancement local :  uvicorn app.main:app --reload
FICHIER PROTÉGÉ : il porte l'authentification et les en-têtes de sécurité.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

from . import VERSION, api, audit, db, veille

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
    "font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; "
    "form-action 'self'"
)
NO_CACHE_PATHS = {"/", "/index.html", "/app.js", "/style.css", "/sw.js", "/manifest.webmanifest"}


class SecurityHeaders:
    """Middleware ASGI pur (compatible avec le flux SSE du chat)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                extra = {
                    b"content-security-policy": CSP.encode(),
                    b"x-content-type-options": b"nosniff",
                    b"referrer-policy": b"no-referrer",
                    b"x-frame-options": b"DENY",
                    b"permissions-policy": b"camera=(), microphone=(), geolocation=()",
                }
                if path in NO_CACHE_PATHS:
                    extra[b"cache-control"] = b"no-cache"
                if path.startswith("/api/") and path != "/api/files" and not path.startswith("/api/files/"):
                    extra[b"cache-control"] = b"no-store"
                present = {k.lower() for k, _ in headers}
                headers += [(k, v) for k, v in extra.items() if k not in present]
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


@asynccontextmanager
async def lifespan(app):
    logging.basicConfig(level=logging.INFO)
    db.init_db()
    veille.seed_defaults()
    audit.log("startup", {"version": VERSION})
    yield


app = Starlette(
    routes=[*api.routes(), Mount("/", app=StaticFiles(directory=str(WEB_DIR), html=True), name="web")],
    middleware=[Middleware(SecurityHeaders)],
    lifespan=lifespan,
)
