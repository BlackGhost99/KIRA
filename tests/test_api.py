"""API HTTP : accès, chat en flux, mémoire, veille, évolution, fichiers, export."""
import asyncio
import base64
import json
import time
import unittest
from datetime import datetime, timezone
from unittest import mock

from app import audit, auth, config, db, evolution, jobs, memory, veille
from app.llm.base import LLMError
from app.main import app
from tests.helpers import FakeResponse, KiraTestCase, ScriptedProvider, call, text_result, tool_result, use_router

RSS = """<?xml version="1.0"?><rss version="2.0"><channel><item><title>Un article utile</title><link>https://exemple.org/a</link>
<description>Une explication détaillée du principe de moindre action en mécanique lagrangienne, avec des exemples concrets sur le pendule et la chute libre pour bien comprendre.</description></item></channel></rss>"""


def wait_jobs(timeout: float = 30):
    end = time.time() + timeout
    while jobs.status()["running"] and time.time() < end:
        time.sleep(0.02)
    assert not jobs.status()["running"], "la tâche de fond ne se termine pas"


class ApiTestCase(KiraTestCase):
    def setUp(self):
        super().setUp()
        self.token = auth.make_token()

    def api(self, method, path, body=None, token="default", **kw):
        headers = dict(kw.pop("headers", {}))
        tok = self.token if token == "default" else token
        if tok:
            headers["authorization"] = f"Bearer {tok}"
        return call(app, method, path, json_body=body, headers=headers, **kw)


class AccessTests(ApiTestCase):
    def test_health_is_public_and_security_headers_are_set(self):
        r = call(app, "GET", "/healthz")
        self.assertEqual((r.status, r.json()["ok"]), (200, True))
        self.assertIn("default-src 'self'", r.headers["content-security-policy"])
        self.assertEqual(r.headers["x-content-type-options"], "nosniff")
        self.assertEqual(r.headers["x-frame-options"], "DENY")

    def test_api_responses_are_not_cached_and_shell_is_revalidated(self):
        self.assertEqual(self.api("GET", "/api/auth/me").headers["cache-control"], "no-store")
        self.assertEqual(call(app, "GET", "/").headers["cache-control"], "no-cache")

    def test_every_owner_route_refuses_anonymous_and_forged_tokens(self):
        protected = [
            ("GET", "/api/auth/me"), ("GET", "/api/status"), ("GET", "/api/briefing"), ("GET", "/api/conversations"),
            ("GET", "/api/conversations/x"), ("PATCH", "/api/conversations/x"), ("DELETE", "/api/conversations/x"),
            ("POST", "/api/chat"), ("POST", "/api/messages/1/feedback"), ("GET", "/api/memory"), ("POST", "/api/memory"),
            ("PATCH", "/api/memory/1"), ("DELETE", "/api/memory/1"), ("GET", "/api/veille/items"),
            ("POST", "/api/veille/items/1/validate"), ("POST", "/api/veille/items/1/reject"),
            ("POST", "/api/veille/validate-all"), ("GET", "/api/veille/sources"), ("POST", "/api/veille/sources"),
            ("PATCH", "/api/veille/sources/1"), ("DELETE", "/api/veille/sources/1"), ("POST", "/api/veille/run"),
            ("GET", "/api/evolution"), ("POST", "/api/evolution/propose"), ("POST", "/api/evolution/1/approve"),
            ("POST", "/api/evolution/1/reject"), ("GET", "/api/audit"), ("GET", "/api/export"), ("POST", "/api/cron/daily"),
        ]
        for method, path in protected:
            for token in (None, "faux.jeton", auth.make_token(days=-1)):
                with self.subTest(route=f"{method} {path}", token=token):
                    self.assertEqual(self.api(method, path, {} if method in ("POST", "PATCH") else None, token=token).status, 401)

    def test_login_success_failure_and_throttle(self):
        bad = call(app, "POST", "/api/auth/login", {"password": "faux"})
        self.assertEqual(bad.status, 401)
        ok = call(app, "POST", "/api/auth/login", {"password": "mot-de-passe-test"})
        self.assertEqual(ok.status, 200)
        token = ok.json()["token"]
        self.assertEqual(self.api("GET", "/api/auth/me", token=token).json()["name"], "Brice")
        self.assertIn("login", {e["action"] for e in audit.recent()})
        for _ in range(6):
            call(app, "POST", "/api/auth/login", {"password": "faux"})
        blocked = call(app, "POST", "/api/auth/login", {"password": "mot-de-passe-test"})
        self.assertEqual(blocked.status, 429)
        auth.throttle.ok("203.0.113.9")
        auth._global = None

    def test_login_without_configured_password(self):
        config.settings.owner_password = ""
        self.assertEqual(call(app, "POST", "/api/auth/login", {"password": ""}).status, 503)

    def test_bad_bodies(self):
        self.assertEqual(self.api("POST", "/api/chat", raw_body=b"pas du json").status, 400)
        self.assertEqual(self.api("POST", "/api/chat", raw_body=b"[1, 2]").status, 400)
        self.assertEqual(self.api("POST", "/api/memory", {"content": 42}).status, 400)
        self.assertEqual(self.api("POST", "/api/chat", {"message": "x" * 25000}).status, 400)


class ChatTests(ApiTestCase):
    def test_chat_stream_conversation_lifecycle(self):
        use_router(ScriptedProvider([tool_result("python", {"code": "print(21*2)"}), text_result("**42**, vérifié."), text_result("Avec plaisir.")]))
        r = self.api("POST", "/api/chat", {"message": "Combien font 21 fois 2 ?"})
        self.assertEqual(r.status, 200)
        self.assertEqual(r.headers["content-type"].split(";")[0], "text/event-stream")
        events = r.events()
        self.assertEqual([e["type"] for e in events], ["conversation", "user_saved", "status", "message", "end"])
        self.assertEqual(events[2]["text"], "Calcul en Python")
        self.assertEqual(events[3]["content"], "**42**, vérifié.")
        conv_id = events[0]["id"]

        again = self.api("POST", "/api/chat", {"message": "Merci", "conversation_id": conv_id}).events()
        self.assertEqual(again[0]["id"], conv_id)

        listing = self.api("GET", "/api/conversations").json()["conversations"]
        self.assertEqual([c["id"] for c in listing], [conv_id])
        detail = self.api("GET", f"/api/conversations/{conv_id}").json()
        self.assertEqual([m["role"] for m in detail["messages"]], ["user", "assistant", "user", "assistant"])
        self.assertEqual(detail["messages"][1]["meta"]["tools"], ["python"])

        self.assertEqual(self.api("PATCH", f"/api/conversations/{conv_id}", {"title": "Mon calcul"}).status, 200)
        self.assertEqual(self.api("PATCH", f"/api/conversations/{conv_id}", {"title": "  "}).status, 400)
        self.assertEqual(self.api("DELETE", f"/api/conversations/{conv_id}").status, 200)
        self.assertEqual(self.api("GET", f"/api/conversations/{conv_id}").status, 404)
        self.assertEqual(db.get_db().q("SELECT * FROM messages"), [])

    def test_reply_is_saved_even_if_the_client_drops_the_connection(self):
        use_router(ScriptedProvider([text_result("Réponse bien enregistrée.")]))
        body = json.dumps({"message": "Salut"}).encode()
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": "/api/chat", "raw_path": b"/api/chat", "query_string": b"", "root_path": "", "client": ("203.0.113.9", 5000),
            "server": ("testserver", 80),
            "headers": [(b"authorization", f"Bearer {self.token}".encode()), (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode())],
        }

        async def run():
            first = True

            async def receive():
                nonlocal first
                if first:
                    first = False
                    return {"type": "http.request", "body": body, "more_body": False}
                return {"type": "http.disconnect"}  # le téléphone a coupé tout de suite

            async def send(message):
                if message["type"] == "http.response.body" and message.get("body"):
                    await asyncio.sleep(0)

            await app(scope, receive, send)

        asyncio.run(run())
        end = time.time() + 15
        rows = []
        while time.time() < end and not rows:
            rows = db.get_db().q("SELECT content FROM messages WHERE role = 'assistant'")
            time.sleep(0.05)
        self.assertEqual([r["content"] for r in rows], ["Réponse bien enregistrée."])

    def test_chat_without_any_ai_returns_an_actionable_error_event(self):
        events = self.api("POST", "/api/chat", {"message": "bonjour"}).events()
        self.assertEqual(events[-2]["type"], "error")
        self.assertIn("ANTHROPIC_API_KEY", events[-2]["message"])
        self.assertEqual(events[-1]["type"], "end")

    def test_chat_deep_tier_and_invalid_tier(self):
        provider = ScriptedProvider(["a", "b"])
        use_router(provider)
        self.api("POST", "/api/chat", {"message": "x", "tier": "deep"})
        self.api("POST", "/api/chat", {"message": "y", "tier": "n'importe quoi"})
        self.assertEqual([c["tier"] for c in provider.calls], ["deep", "default"])

    def test_feedback(self):
        use_router(ScriptedProvider(["Réponse"]))
        events = self.api("POST", "/api/chat", {"message": "q"}).events()
        mid = events[-2]["id"]
        self.assertEqual(self.api("POST", f"/api/messages/{mid}/feedback", {"value": -1, "note": "trop vague"}).status, 200)
        row = db.get_db().q1("SELECT feedback, feedback_note FROM messages WHERE id = ?", [mid])
        self.assertEqual((row["feedback"], row["feedback_note"]), (-1, "trop vague"))
        self.api("POST", f"/api/messages/{mid}/feedback", {"value": 0})
        self.assertIsNone(db.get_db().q1("SELECT feedback FROM messages WHERE id = ?", [mid])["feedback"])
        self.assertEqual(self.api("POST", f"/api/messages/{mid}/feedback", {"value": 5}).status, 400)
        self.assertEqual(self.api("POST", "/api/messages/99999/feedback", {"value": 1}).status, 404)
        user_msg = events[1]["id"]
        self.assertEqual(self.api("POST", f"/api/messages/{user_msg}/feedback", {"value": 1}).status, 404)

    def test_generated_plot_is_private_and_never_cached(self):
        use_router(ScriptedProvider([tool_result("python", {"code": "import matplotlib.pyplot as plt\nplt.plot([1,2])\nplt.show()"}), "Voici."]))
        events = self.api("POST", "/api/chat", {"message": "trace"}).events()
        files = events[-2]["files"]
        self.assertEqual(len(files), 1)
        self.assertEqual(call(app, "GET", files[0]["url"]).status, 401)  # jamais public : captures d'écran, graphiques privés
        r = self.api("GET", files[0]["url"])
        self.assertEqual(r.status, 200)
        self.assertTrue(r.body.startswith(b"\x89PNG"))
        self.assertEqual(r.headers["cache-control"], "private, no-store")
        self.assertEqual(self.api("GET", "/api/files/" + "0" * 32).status, 404)
        self.assertEqual(self.api("GET", "/api/files/../../etc/passwd").status, 404)


class StatusTests(ApiTestCase):
    def test_status_and_briefing(self):
        use_router(ScriptedProvider(["x"]))
        s = self.api("GET", "/api/status").json()
        self.assertEqual(s["owner"], "Brice")
        self.assertEqual(s["database"], "sqlite")
        self.assertEqual(s["providers"][0]["name"], "fake")
        self.assertEqual(s["budget"]["limit"], config.settings.daily_token_budget)
        self.assertIn("python", s["tools"])
        self.assertFalse(s["github"])
        memory.add("profile", "Brice veut comprendre la relativité")
        for hour, expected in ((2, "Encore debout, Brice ?"), (10, "Bonjour Brice"),
                               (15, "Bon après-midi Brice"), (20, "Bonsoir Brice")):
            with self.subTest(hour=hour), mock.patch("app.briefing.local_now", return_value=datetime(2026, 10, 7, hour, tzinfo=timezone.utc)):
                b = self.api("GET", "/api/briefing").json()
                self.assertEqual(b["greeting"], expected)
        self.assertEqual(b["known"], 1)
        self.assertTrue(any("relativité" in x for x in b["suggestions"]))
        self.assertLessEqual(len(b["suggestions"]), 4)

    def test_audit_and_export(self):
        audit.log("test_action", {"x": 1})
        entries = self.api("GET", "/api/audit", query="prefix=test_").json()["entries"]
        self.assertEqual(entries[0]["details"], {"x": 1})
        memory.add("fact", "un souvenir")
        r = self.api("GET", "/api/export")
        self.assertIn("attachment", r.headers["content-disposition"])
        dump = r.json()
        self.assertEqual(dump["memories"][0]["content"], "un souvenir")
        self.assertIn("conversations", dump)
        self.assertNotIn("mot-de-passe-test", r.text)


class MemoryApiTests(ApiTestCase):
    def test_crud_and_search(self):
        created = self.api("POST", "/api/memory", {"kind": "profile", "content": "Brice aime l'astronomie", "tags": "loisirs"}).json()
        mid = created["id"]
        self.assertEqual(self.api("GET", "/api/memory").json()["counts"], {"profile": 1})
        self.assertEqual(self.api("GET", "/api/memory", query="q=astronomie").json()["items"][0]["id"], mid)
        self.assertEqual(self.api("GET", "/api/memory", query="kind=fact").json()["items"], [])
        patched = self.api("PATCH", f"/api/memory/{mid}", {"content": "Brice adore l'astronomie", "kind": "fact"}).json()
        self.assertEqual((patched["content"], patched["kind"]), ("Brice adore l'astronomie", "fact"))
        self.assertEqual(self.api("POST", "/api/memory", {"content": "   "}).status, 400)
        self.assertEqual(self.api("PATCH", "/api/memory/999", {"content": "x"}).status, 404)
        self.assertEqual(self.api("DELETE", f"/api/memory/{mid}").status, 200)
        self.assertEqual(self.api("DELETE", f"/api/memory/{mid}").status, 404)


class VeilleApiTests(ApiTestCase):
    JSON_OK = json.dumps({"resume": "Résumé court.", "sujet": "physique", "niveau": "débutant", "pertinence": 1})

    def setUp(self):
        super().setUp()
        db.get_db().run("DELETE FROM veille_sources")
        self.http.route("GET", "exemple.org/feed", FakeResponse(200, body=RSS.encode(), headers={"content-type": "application/rss+xml"}))

    def test_sources_and_run_and_validation_flow(self):
        added = self.api("POST", "/api/veille/sources", {"url": "https://exemple.org/feed", "name": "Test", "score": 0.8}).json()
        self.assertEqual(self.api("POST", "/api/veille/sources", {"url": "ftp://non"}).status, 400)
        self.assertEqual(self.api("GET", "/api/veille/sources").json()["sources"][0]["name"], "Test")
        use_router(ScriptedProvider([self.JSON_OK]))
        run = self.api("POST", "/api/veille/run")
        self.assertEqual((run.status, run.json()["started"]), (202, True))
        wait_jobs()
        self.assertTrue(jobs.status()["last"]["veille"]["ok"])
        items = self.api("GET", "/api/veille/items").json()
        self.assertEqual(items["counts"], {"pending": 1})
        item = items["items"][0]
        self.assertEqual(item["score"], 0.8)
        self.assertEqual(self.api("POST", f"/api/veille/items/{item['id']}/validate").json()["status"], "validated")
        self.assertEqual(memory.counts(), {"knowledge": 1})
        self.assertEqual(self.api("GET", "/api/veille/items", query="status=validated").json()["items"][0]["id"], item["id"])
        self.assertEqual(self.api("GET", "/api/veille/items", query="status=bizarre").status, 400)
        self.assertEqual(self.api("POST", "/api/veille/items/999/validate").status, 404)
        self.assertEqual(self.api("PATCH", f"/api/veille/sources/{added['id']}", {"active": False}).json()["active"], 0)
        self.assertEqual(self.api("DELETE", f"/api/veille/sources/{added['id']}").status, 200)
        self.assertEqual(self.api("DELETE", f"/api/veille/sources/{added['id']}").status, 404)

    def test_reject_and_validate_all(self):
        veille.add_source("https://exemple.org/feed", "Test", "rss", 0.9)
        use_router(ScriptedProvider([self.JSON_OK]))
        veille.run()
        item = veille.list_items()[0]
        self.assertEqual(self.api("POST", "/api/veille/validate-all", {"min_score": 0.5}).json()["validated"], 1)
        self.assertEqual(self.api("POST", f"/api/veille/items/{item['id']}/reject").json()["status"], "rejected")
        self.assertEqual(self.api("POST", "/api/veille/validate-all", {"min_score": "abc"}).status, 400)

    def test_cron_daily_accepts_cron_token_and_runs_veille_and_consolidation(self):
        veille.add_source("https://exemple.org/feed", "Test", "rss", 0.9)
        use_router(ScriptedProvider([self.JSON_OK]))
        self.assertEqual(call(app, "POST", "/api/cron/daily").status, 401)
        self.assertEqual(call(app, "POST", "/api/cron/daily", headers={"x-cron-token": "mauvais"}).status, 401)
        ok = call(app, "POST", "/api/cron/daily", headers={"x-cron-token": "jeton-cron-test"})
        self.assertEqual((ok.status, ok.json()), (202, {"started": True}))
        wait_jobs()
        result = jobs.status()["last"]["daily"]["result"]
        self.assertEqual(result["veille"]["new"], 1)
        self.assertIn("skipped", result["consolidation"])

    def test_second_job_while_running_is_refused(self):
        import threading
        gate = threading.Event()
        self.assertTrue(jobs.start("lent", lambda: gate.wait(5) or {}))
        self.assertFalse(jobs.start("lent", lambda: {}))
        gate.set()
        wait_jobs()
        self.assertTrue(jobs.start("lent", lambda: {}, wait=True))


class EvolutionApiTests(ApiTestCase):
    FIND = "pas de remplissage."
    PROPOSAL = {"proposal": {"title": "Ton plus direct", "rationale": "Trop bavard.", "changes": [
        {"path": "app/agent.py", "action": "edit", "find": FIND, "replace": "pas de remplissage. Pas de flatterie."}]}}

    def setUp(self):
        super().setUp()
        config.settings.evolution_run_tests = False

    def test_propose_list_approve_flow(self):
        use_router(ScriptedProvider(["```json\n" + json.dumps(self.PROPOSAL) + "\n```"]))
        r = self.api("POST", "/api/evolution/propose", {"goal": "Sois plus direct"})
        self.assertEqual(r.status, 202)
        wait_jobs()
        proposals = self.api("GET", "/api/evolution").json()["proposals"]
        self.assertEqual(len(proposals), 1)
        pid = proposals[0]["id"]
        self.assertEqual(proposals[0]["status"], "draft")
        self.assertIn("Pas de flatterie", proposals[0]["diff"])
        self.assertEqual(self.api("GET", "/api/status").json()["proposals_draft"], 1)
        # sans jeton GitHub : message clair, la proposition reste à l'état de brouillon
        refused = self.api("POST", f"/api/evolution/{pid}/approve")
        self.assertEqual(refused.status, 400)
        self.assertIn("GITHUB_TOKEN", refused.json()["error"])
        # avec jeton : la pull request est ouverte
        config.settings.github_token = "ghp_test"
        source = evolution.read_local("app/agent.py")
        repo = "/repos/BlackGhost99/KIRA"
        self.http.route("GET", f"{repo}/contents/app/agent.py", FakeResponse(200, {"sha": "s1", "content": base64.b64encode(source.encode()).decode()}))
        self.http.route("GET", f"{repo}/git/ref/heads/main", FakeResponse(200, {"object": {"sha": "b1"}}))
        self.http.route("POST", f"{repo}/git/refs", FakeResponse(201, {}))
        self.http.route("PUT", f"{repo}/contents/app/agent.py", FakeResponse(200, {}))
        self.http.route("POST", f"{repo}/pulls", FakeResponse(201, {"html_url": "https://github.com/BlackGhost99/KIRA/pull/1"}))
        done = self.api("POST", f"/api/evolution/{pid}/approve").json()
        self.assertEqual((done["status"], done["pr_url"]), ("pr_opened", "https://github.com/BlackGhost99/KIRA/pull/1"))

    def test_reject_and_unknown(self):
        use_router(ScriptedProvider(["```json\n" + json.dumps(self.PROPOSAL) + "\n```"]))
        self.api("POST", "/api/evolution/propose", {})
        wait_jobs()
        pid = self.api("GET", "/api/evolution").json()["proposals"][0]["id"]
        self.assertEqual(self.api("POST", f"/api/evolution/{pid}/reject").json()["status"], "rejected")
        self.assertEqual(self.api("POST", "/api/evolution/999/reject").status, 404)
        self.assertEqual(self.api("POST", "/api/evolution/999/approve").status, 400)

    def test_failed_analysis_is_visible_in_job_status(self):
        use_router(ScriptedProvider([LLMError("panne", 500, "x")]))
        self.api("POST", "/api/evolution/propose", {})
        wait_jobs()
        last = self.api("GET", "/api/evolution").json()["jobs"]["last"]["evolution"]
        self.assertFalse(last["ok"])


if __name__ == "__main__":
    unittest.main()
