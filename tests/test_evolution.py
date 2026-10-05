"""Évolution : périmètre, application des changements, vérification, pull request."""
import base64
import json
import os
import unittest

from app import config, db, evolution
from app.evolution import EvolutionError, apply_changes, check_path, make_diff, read_local
from app.principles import PROTECTED_PATHS
from tests.helpers import FakeResponse, KiraTestCase, ScriptedProvider, use_router

FIND = "pas de remplissage."  # une seule fois dans app/agent.py ; le remplacement le conserve, donc les tests
# restent valides quand la suite est relancée sur une copie où le changement est appliqué
GOOD = {"proposal": {"title": "Ton plus direct", "rationale": "Les réponses sont trop bavardes.", "risks": "Aucun.",
                     "changes": [{"path": "app/agent.py", "action": "edit", "find": FIND, "replace": "pas de remplissage. Pas de flatterie."}]}}


def as_reply(obj) -> str:
    return "Voici ma proposition :\n```json\n" + json.dumps(obj, ensure_ascii=False) + "\n```"


class PerimeterTests(unittest.TestCase):
    def test_protected_files_exist(self):
        for path in PROTECTED_PATHS:
            self.assertIsNotNone(read_local(path), f"{path} est déclaré protégé mais n'existe pas")
        self.assertIn("app/evolution.py", PROTECTED_PATHS)
        self.assertIn("app/principles.py", PROTECTED_PATHS)

    def test_check_path(self):
        self.assertIsNone(check_path("app/agent.py"))
        self.assertIsNone(check_path("web/app.js"))
        self.assertIsNone(check_path("tests/test_nouveau.py"))
        for bad in ("app/auth.py", "app/principles.py", "app/evolution.py", "app/sandbox.py", "app/net.py", "app/budget.py",
                    "app/main.py", "app/devices.py", "app/tools/device_tools.py", "core/kira_core.py", "tests/test_devices.py",
                    "tests/test_core_agent.py", "Dockerfile", "render.yaml", ".github/workflows/ci.yml", "requirements.txt",
                    "../etc/passwd", "app/../Dockerfile", "web/vendor/marked.js", "app/données.db", "", "legacy/kira_v1/core.py"):
            with self.subTest(path=bad):
                self.assertIsNotNone(check_path(bad))


class ApplyChangesTests(unittest.TestCase):
    def read(self, path):
        return {"app/a.py": "x = 1\ny = 2\nx = 1 + y\n", "app/b.py": "def f():\n    return 1\n"}.get(path)

    def test_edit_and_create(self):
        files, errors = apply_changes([
            {"path": "app/a.py", "action": "edit", "find": "y = 2", "replace": "y = 3"},
            {"path": "app/c.py", "action": "create", "content": "z = 0\n"},
        ], self.read)
        self.assertEqual(errors, [])
        self.assertEqual(files["app/a.py"], "x = 1\ny = 3\nx = 1 + y\n")
        self.assertEqual(files["app/c.py"], "z = 0\n")

    def test_sequential_edits_on_same_file(self):
        files, errors = apply_changes([
            {"path": "app/a.py", "action": "edit", "find": "y = 2", "replace": "y = 3"},
            {"path": "app/a.py", "action": "edit", "find": "y = 3", "replace": "y = 4"},
        ], self.read)
        self.assertEqual(errors, [])
        self.assertIn("y = 4", files["app/a.py"])

    def test_ambiguous_missing_and_malformed_edits(self):
        _, errors = apply_changes([{"path": "app/a.py", "action": "edit", "find": "x = 1", "replace": "x = 9"}], self.read)
        self.assertIn("2 fois", errors[0])
        _, errors = apply_changes([{"path": "app/a.py", "action": "edit", "find": "absent", "replace": "x"}], self.read)
        self.assertIn("0 fois", errors[0])
        _, errors = apply_changes([{"path": "app/zzz.py", "action": "edit", "find": "a", "replace": "b"}], self.read)
        self.assertIn("n'existe pas", errors[0])
        _, errors = apply_changes([{"path": "app/a.py", "action": "edit", "find": "", "replace": "b"}], self.read)
        self.assertTrue(errors)
        _, errors = apply_changes([{"path": "app/a.py", "action": "delete"}], self.read)
        self.assertIn("action inconnue", errors[0])
        for junk in (None, [], "texte", [42]):
            self.assertTrue(apply_changes(junk, self.read)[1])

    def test_create_over_existing_and_protected_target(self):
        _, errors = apply_changes([{"path": "app/a.py", "action": "create", "content": "x"}], self.read)
        self.assertIn("existe déjà", errors[0])
        _, errors = apply_changes([{"path": "app/auth.py", "action": "edit", "find": "a", "replace": "b"}], self.read)
        self.assertIn("protégé", errors[0])

    def test_syntax_error_is_caught(self):
        _, errors = apply_changes([{"path": "app/a.py", "action": "edit", "find": "y = 2", "replace": "y = = 2"}], self.read)
        self.assertIn("erreur de syntaxe", errors[0])

    def test_too_many_files(self):
        changes = [{"path": f"app/n{i}.py", "action": "create", "content": "a = 1\n"} for i in range(evolution.MAX_FILES + 1)]
        self.assertIn("trop de fichiers", apply_changes(changes, self.read)[1][0])

    def test_diff_shows_changes(self):
        files, _ = apply_changes([{"path": "app/a.py", "action": "edit", "find": "y = 2", "replace": "y = 3"}], self.read)
        diff = make_diff(files, self.read)
        self.assertIn("--- a/app/a.py", diff)
        self.assertIn("-y = 2", diff)
        self.assertIn("+y = 3", diff)


class ContextTests(KiraTestCase):
    def test_snapshot_has_real_code_and_excludes_vendor_and_legacy(self):
        snap = evolution.source_snapshot()
        self.assertIn("=== app/agent.py ===", snap)
        self.assertIn("def run_turn", snap)
        headers = [line for line in snap.splitlines() if line.startswith("=== ")]
        self.assertTrue(headers)
        for header in headers:
            self.assertFalse(header.startswith(("=== legacy/", "=== web/vendor/")), header)
            self.assertNotIn("__pycache__", header)

    def test_usage_report_includes_bad_answers_and_previous_proposals(self):
        d = db.get_db()
        d.run("INSERT INTO conversations (id, title, created_at, updated_at) VALUES ('c','t','x','x')")
        d.insert("messages", {"conversation_id": "c", "role": "user", "content": "Explique le spin", "meta": "{}", "created_at": "1"})
        d.insert("messages", {"conversation_id": "c", "role": "assistant", "content": "Réponse floue", "meta": "{}", "feedback": -1,
                              "feedback_note": "trop vague", "created_at": "2"})
        d.insert("proposals", {"title": "Ancienne idée", "status": "rejected", "created_at": "x", "updated_at": "x"})
        report = evolution.usage_report()
        self.assertIn("Explique le spin", report)
        self.assertIn("trop vague", report)
        self.assertIn("Ancienne idée [rejected]", report)


class ProposeTests(KiraTestCase):
    def setUp(self):
        super().setUp()
        config.settings.evolution_run_tests = False

    def test_valid_proposal_is_stored_with_diff(self):
        provider = ScriptedProvider([as_reply(GOOD)])
        use_router(provider)
        result = evolution.propose("Rends les réponses plus directes")
        self.assertEqual(result["status"], "created")
        proposal = evolution.get_proposal(result["id"])
        self.assertEqual(proposal["status"], "draft")
        self.assertEqual(proposal["files"], ["app/agent.py"])
        self.assertIn("+- Réponds en français", proposal["diff"])
        self.assertIn("Pas de flatterie", proposal["diff"])
        self.assertIsNone(proposal["tests_ok"])
        self.assertEqual(provider.calls[0]["tier"], "deep")
        prompt = provider.calls[0]["messages"][0]["content"]
        self.assertIn("Rends les réponses plus directes", prompt)
        self.assertIn("def run_turn", prompt)  # le vrai code est fourni

    def test_retries_with_feedback_until_valid(self):
        bad_find = {"proposal": {**GOOD["proposal"], "changes": [{"path": "app/agent.py", "action": "edit", "find": "texte inexistant", "replace": "x"}]}}
        provider = ScriptedProvider(["je ne sais pas faire du JSON", as_reply(bad_find), as_reply(GOOD)])
        use_router(provider)
        self.assertEqual(evolution.propose()["status"], "created")
        self.assertEqual(len(provider.calls), 3)
        self.assertIn("JSON", provider.calls[1]["messages"][-1]["content"])
        self.assertIn("0 fois", provider.calls[2]["messages"][-1]["content"])

    def test_protected_file_proposals_are_never_stored(self):
        evil = {"proposal": {"title": "Je me libère", "rationale": "x", "changes": [
            {"path": "app/principles.py", "action": "edit", "find": "PRINCIPES", "replace": "AUTRE"}]}}
        use_router(ScriptedProvider([as_reply(evil)] * 3))
        with self.assertRaises(EvolutionError) as ctx:
            evolution.propose()
        self.assertIn("protégé", str(ctx.exception))
        self.assertEqual(db.get_db().q("SELECT * FROM proposals"), [])

    def test_no_proposal_is_a_valid_answer(self):
        use_router(ScriptedProvider([as_reply({"proposal": None, "reason": "Rien de nécessaire."})]))
        self.assertEqual(evolution.propose(), {"status": "none", "reason": "Rien de nécessaire."})

    def test_failing_tests_trigger_a_retry_then_pass(self):
        outputs = iter([(False, "FAIL: test_x"), (True, "OK")])
        evolution_run = evolution.run_tests
        evolution.run_tests = lambda files: next(outputs)
        try:
            provider = ScriptedProvider([as_reply(GOOD), as_reply(GOOD)])
            use_router(provider)
            result = evolution.propose()
        finally:
            evolution.run_tests = evolution_run
        self.assertEqual(evolution.get_proposal(result["id"])["tests_ok"], True)
        self.assertIn("FAIL: test_x", provider.calls[1]["messages"][-1]["content"])

    def test_persistently_failing_tests_are_stored_but_cannot_be_approved(self):
        evolution_run = evolution.run_tests
        evolution.run_tests = lambda files: (False, "FAIL")
        try:
            use_router(ScriptedProvider([as_reply(GOOD)] * 3))
            result = evolution.propose()
        finally:
            evolution.run_tests = evolution_run
        self.assertFalse(evolution.get_proposal(result["id"])["tests_ok"])
        config.settings.github_token = "ghp_test"
        with self.assertRaises(EvolutionError) as ctx:
            evolution.approve(result["id"])
        self.assertIn("tests échouent", str(ctx.exception))
        self.assertEqual(self.http.calls, [])


@unittest.skipIf(os.environ.get("KIRA_NESTED_TESTS"), "déjà dans une exécution de tests imbriquée")
class RealTestRunTests(KiraTestCase):
    def test_passing_change_is_validated_by_the_real_suite(self):
        files, errors = apply_changes(GOOD["proposal"]["changes"], read_local)
        self.assertEqual(errors, [])
        ok, output = evolution.run_tests(files)
        self.assertTrue(ok, output)

    def test_breaking_change_is_caught_by_the_real_suite(self):
        broken = {"tests/test_zz_casse.py": "import unittest\n\nclass T(unittest.TestCase):\n    def test_casse(self):\n        self.assertEqual(1, 2)\n"}
        ok, output = evolution.run_tests(broken)
        self.assertFalse(ok)
        self.assertIn("test_casse", output)

    def test_tests_disabled_by_setting(self):
        config.settings.evolution_run_tests = False
        self.assertIsNone(evolution.run_tests({})[0])


class PullRequestTests(KiraTestCase):
    REPO = "BlackGhost99/KIRA"

    def setUp(self):
        super().setUp()
        config.settings.evolution_run_tests = False
        config.settings.github_token = "ghp_test"
        use_router(ScriptedProvider([as_reply(GOOD)]))
        self.pid = evolution.propose()["id"]
        source = read_local("app/agent.py")
        base = f"/repos/{self.REPO}"
        self.http.route("GET", f"{base}/contents/app/agent.py", FakeResponse(200, {"sha": "sha-agent", "content": base64.b64encode(source.encode()).decode()}))
        self.http.route("GET", f"{base}/git/ref/heads/main", FakeResponse(200, {"object": {"sha": "base-sha"}}))
        self.http.route("POST", f"{base}/git/refs", FakeResponse(201, {}))
        self.http.route("PUT", f"{base}/contents/app/agent.py", FakeResponse(200, {}))
        self.http.route("POST", f"{base}/pulls", FakeResponse(201, {"html_url": f"https://github.com/{self.REPO}/pull/7"}))

    def test_approval_opens_a_pull_request_and_never_merges(self):
        proposal = evolution.approve(self.pid)
        self.assertEqual(proposal["status"], "pr_opened")
        self.assertEqual(proposal["pr_url"], f"https://github.com/{self.REPO}/pull/7")
        self.assertTrue(proposal["branch"].startswith(f"kira/evolution-{self.pid}-ton-plus-direct"))
        calls = {(c["method"], c["url"].split("github.com")[1]): c for c in self.http.calls}
        ref = calls[("POST", f"/repos/{self.REPO}/git/refs")]["json"]
        self.assertEqual(ref["sha"], "base-sha")
        put = calls[("PUT", f"/repos/{self.REPO}/contents/app/agent.py")]["json"]
        self.assertEqual(put["sha"], "sha-agent")
        self.assertEqual(put["branch"], proposal["branch"])
        self.assertIn("Pas de flatterie", base64.b64decode(put["content"]).decode())
        pr = calls[("POST", f"/repos/{self.REPO}/pulls")]["json"]
        self.assertEqual(pr["base"], "main")
        self.assertIn("Rien n'est déployé", pr["body"])
        self.assertFalse([c for c in self.http.calls if "merge" in c["url"]])
        self.assertTrue(all(c["headers"]["Authorization"] == "Bearer ghp_test" for c in self.http.calls))

    def test_cannot_approve_twice_and_reject_works(self):
        evolution.approve(self.pid)
        with self.assertRaises(EvolutionError):
            evolution.approve(self.pid)
        use_router(ScriptedProvider([as_reply(GOOD)]))
        other = evolution.propose()["id"]
        self.assertEqual(evolution.reject(other)["status"], "rejected")
        with self.assertRaises(EvolutionError):
            evolution.approve(other)

    def test_missing_token_gives_actionable_message_and_keeps_draft(self):
        config.settings.github_token = ""
        with self.assertRaises(EvolutionError) as ctx:
            evolution.approve(self.pid)
        self.assertIn("GITHUB_TOKEN", str(ctx.exception))
        self.assertEqual(evolution.get_proposal(self.pid)["status"], "draft")

    def test_repo_changed_since_proposal_is_reported(self):
        self.http.routes = [r for r in self.http.routes if not (r[0] == "GET" and "contents/app/agent.py" in r[1])]
        self.http.route("GET", f"/repos/{self.REPO}/contents/app/agent.py", FakeResponse(200, {"sha": "s", "content": base64.b64encode(b"x = 1\n").decode()}))
        with self.assertRaises(EvolutionError) as ctx:
            evolution.approve(self.pid)
        self.assertIn("Le dépôt a changé", str(ctx.exception))
        proposal = evolution.get_proposal(self.pid)
        self.assertEqual(proposal["status"], "draft")
        self.assertIn("Le dépôt a changé", proposal["error"])

    def test_github_failure_on_branch_creation(self):
        self.http.routes = [r for r in self.http.routes if not (r[0] == "POST" and r[1].endswith("/git/refs"))]
        self.http.route("POST", f"/repos/{self.REPO}/git/refs", FakeResponse(422, text="Reference already exists"))
        with self.assertRaises(EvolutionError) as ctx:
            evolution.approve(self.pid)
        self.assertIn("HTTP 422", str(ctx.exception))
        self.assertFalse([c for c in self.http.calls if c["method"] == "PUT"])


if __name__ == "__main__":
    unittest.main()
