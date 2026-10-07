"""Outillage commun aux tests : base temporaire, faux IA, faux HTTP, appel direct de l'application ASGI."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import requests

from app import config, db, jobs, net
from app.llm import router as llm_router
from app.llm.base import LLMResult, Provider, ToolCall


class FakeResponse:
    def __init__(self, status: int = 200, json_data=None, text: str | None = None, headers: dict | None = None,
                 body: bytes | None = None):
        self.status_code = status
        self._json = json_data
        if body is None:
            body = (text if text is not None else json.dumps(json_data or {})).encode("utf-8")
        self._body = body
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}

    @property
    def text(self) -> str:
        return self._body.decode("utf-8", "replace")

    def json(self):
        return self._json if self._json is not None else json.loads(self._body)

    def iter_content(self, size: int = 65536):
        for i in range(0, len(self._body), size):
            yield self._body[i : i + size]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def close(self):
        pass


class FakeSession:
    """Remplace requests.Session : on déclare des routes (méthode, morceau d'URL) -> réponse."""

    def __init__(self):
        self.calls: list[dict] = []
        self.routes: list[tuple[str, str, object]] = []
        self.headers: dict = {}

    def route(self, method: str, url_part: str, response) -> None:
        self.routes.append((method.upper(), url_part, response))

    def request(self, method: str, url: str, **kwargs):
        self.calls.append({"method": method.upper(), "url": url, **kwargs})
        for m, part, response in self.routes:
            if m == method.upper() and part in url:
                return response(url, **kwargs) if callable(response) else response
        raise AssertionError(f"Appel HTTP inattendu : {method} {url}")

    def get(self, url, **kw):
        return self.request("GET", url, **kw)

    def post(self, url, **kw):
        return self.request("POST", url, **kw)

    def put(self, url, **kw):
        return self.request("PUT", url, **kw)


class ScriptedProvider(Provider):
    """Faux fournisseur d'IA : renvoie les éléments du script dans l'ordre (résultat, exception ou fonction)."""

    def __init__(self, script, name: str = "fake"):
        self.name = name
        self.script = list(script)
        self.calls: list[dict] = []

    def configured(self) -> bool:
        return True

    def model_for(self, tier: str) -> str:
        return f"{self.name}-{tier}"

    def complete(self, system, messages, tools, tier, max_tokens):
        self.calls.append(
            {"system": system, "messages": copy.deepcopy(messages), "tools": tools, "tier": tier, "max_tokens": max_tokens}
        )
        item = self.script.pop(0) if self.script else LLMResult(text="(script épuisé)")
        if isinstance(item, Exception):
            raise item
        if callable(item):
            item = item(system, messages, tools)
        if isinstance(item, str):
            item = LLMResult(text=item)
        item.provider = item.provider or self.name
        item.model = item.model or self.model_for(tier)
        item.input_tokens = item.input_tokens or 10
        item.output_tokens = item.output_tokens or 5
        return item


def text_result(text: str) -> LLMResult:
    return LLMResult(text=text)


def tool_result(name: str, args: dict, call_id: str = "call_1", text: str = "") -> LLMResult:
    return LLMResult(text=text, tool_calls=[ToolCall(call_id, name, args)])


def use_router(*providers: Provider) -> llm_router.Router:
    router = llm_router.Router(providers=list(providers))
    llm_router.set_router(router)
    return router


class KiraTestCase(unittest.TestCase):
    """Base de test : base SQLite jetable, réglages vierges, réseau simulé.

    Tout le nettoyage passe par addCleanup : il s'exécute même si un setUp
    plante en cours de route (sinon un patch pourrait fuir vers les tests suivants).
    """

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix="kira_test_")
        self.addCleanup(shutil.rmtree, self._tmp, True)
        self.addCleanup(setattr, config, "settings", config.settings)
        config.settings = config.Settings(
            owner_password="mot-de-passe-test", secret_key="cle-secrete-test", cron_token="jeton-cron-test"
        )
        self.addCleanup(self._close_db)
        db.configure(f"sqlite:///{self._tmp}/test.db")
        self.http = FakeSession()
        self.addCleanup(net.set_session, None)
        net.set_session(self.http)
        self._patch = mock.patch("app.net.assert_public_url", lambda url: None)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(llm_router.set_router, None)
        llm_router.set_router(None)
        jobs._running.clear()
        jobs._last.clear()

    @staticmethod
    def _close_db():
        if db._db is not None:
            db._db.close()
        db._db = None


# -- appel direct de l'application ASGI (sans serveur ni httpx) -------------
class Reply:
    def __init__(self, status: int, headers: dict, body: bytes, raw_headers=()):
        self.status = status
        self.headers = headers
        self.body = body
        from http.cookies import SimpleCookie
        self.cookies = SimpleCookie()
        for key, value in raw_headers:
            if key.lower() == b"set-cookie":
                self.cookies.load(value.decode())

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", "replace")

    def json(self):
        return json.loads(self.body)

    def events(self) -> list[dict]:
        return [json.loads(line[6:]) for line in self.text.splitlines() if line.startswith("data: ")]


def call(app, method: str, path: str, json_body=None, headers: dict | None = None, query: str = "",
         raw_body: bytes | None = None) -> Reply:
    body = raw_body if raw_body is not None else (json.dumps(json_body).encode("utf-8") if json_body is not None else b"")
    hdrs = {k.lower(): v for k, v in (headers or {}).items()}
    if body:
        hdrs.setdefault("content-type", "application/json")
        hdrs["content-length"] = str(len(body))
    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method.upper(), "scheme": "http",
        "path": path, "raw_path": path.encode(), "query_string": query.encode(), "root_path": "",
        "headers": [(k.encode(), v.encode()) for k, v in hdrs.items()], "client": ("203.0.113.9", 5000),
        "server": ("testserver", 80),
    }
    sent: list[dict] = []

    async def run():
        finished = asyncio.Event()
        first = True

        async def receive():
            nonlocal first
            if first:
                first = False
                return {"type": "http.request", "body": body, "more_body": False}
            await finished.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body" and not message.get("more_body"):
                finished.set()

        await app(scope, receive, send)

    asyncio.run(run())
    start = next(m for m in sent if m["type"] == "http.response.start")
    out_headers = {k.decode().lower(): v.decode() for k, v in start["headers"]}
    out_body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return Reply(start["status"], out_headers, out_body, start["headers"])
