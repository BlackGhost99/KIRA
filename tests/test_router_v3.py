"""Bascule gratuite, compatibilité et isolation des erreurs fournisseur."""
from unittest.mock import patch

from app import audit, budget, config, db
from app.llm.base import LLMError, LLMUnavailable, ToolCall, ToolSpec
from app.llm.errors import http_error
from app.llm.openai_compat import convert_messages, gemini_provider, openrouter_provider
from tests.helpers import FakeResponse, KiraTestCase, ScriptedProvider, use_router


class FreeProvider(ScriptedProvider):
    def cost_class(self):
        return "free"


class RouterV3Tests(KiraTestCase):
    def test_free_still_works_after_paid_budget_exhaustion(self):
        config.settings.daily_token_budget = 1
        budget.record("paid", "m", 2, 0)
        paid = ScriptedProvider(["payant"], name="paid")
        free = FreeProvider(["gratuit", "encore"], name="free")
        router = use_router(paid, free)
        self.assertEqual(router.complete([], []).text, "gratuit")
        self.assertEqual(router.complete([], []).text, "encore")
        self.assertEqual(paid.calls, [])
        self.assertEqual(budget.summary()["paid_used"], 2)
        self.assertEqual(budget.summary()["used"], 32)

    def test_paid_failure_tries_free_before_another_paid_provider(self):
        broken = ScriptedProvider([LLMError("secret payload", 402, category="billing")], name="a")
        paid = ScriptedProvider(["paid"], name="b")
        free = FreeProvider(["free"], name="c")
        router = use_router(broken, paid, free)
        self.assertEqual(router.complete([], []).provider, "c")
        self.assertEqual(paid.calls, [])
        self.assertNotIn("secret payload", str(audit.recent()))

    def test_economy_does_not_silently_switch_to_paid(self):
        config.settings.llm_routing_mode = "ECONOMY"
        paid = ScriptedProvider(["paid"], name="a")
        free = FreeProvider([LLMError("quota", 429)], name="b")
        router = use_router(paid, free)
        with self.assertRaises(LLMUnavailable):
            router.complete([], [])
        self.assertEqual(paid.calls, [])
        config.settings.llm_economy_allow_paid = True
        self.assertEqual(router.complete([], []).text, "paid")

    def test_private_mode_never_falls_back_to_an_unapproved_provider(self):
        config.settings.llm_trusted_providers = "a"
        approved = ScriptedProvider([LLMError("down", 500)], name="a")
        unapproved = FreeProvider(["must not be used"], name="b")
        router = use_router(approved, unapproved)
        with self.assertRaises(LLMUnavailable):
            router.complete([], [], private=True)
        self.assertEqual(unapproved.calls, [])
        db.get_db().kv_set("llm_routing_mode", "PRIVATE")
        with self.assertRaises(LLMUnavailable):
            router.complete([], [])
        self.assertEqual(unapproved.calls, [])

    def test_vision_and_tools_are_required_before_sending_context(self):
        text_only = FreeProvider(["text"], name="text")
        text_only.vision = False
        no_tools = FreeProvider(["no tools"], name="no-tools")
        no_tools.supports_tools = False
        compatible = FreeProvider(["vision"], name="vision")
        router = use_router(text_only, no_tools, compatible)
        messages = [{"role": "tool", "images": [{"mime": "image/png", "b64": "fake"}]}]
        tools = [ToolSpec("test", "test", {"type": "object"})]
        self.assertEqual(router.complete([], messages, tools).provider, "vision")
        self.assertEqual(text_only.calls, [])
        self.assertEqual(no_tools.calls, [])

    def test_retry_after_is_respected_and_configuration_errors_wait_for_change(self):
        p = ScriptedProvider([LLMError("limit", 429, retry_after=900), "ok"], name="p")
        router = use_router(p)
        with patch("app.llm.router.time.time", return_value=1000):
            with self.assertRaises(LLMUnavailable):
                router.complete([], [])
            self.assertEqual(router._cool["p"], 1900)
        bad = ScriptedProvider([LLMError("key", 401), "fixed"], name="bad")
        bad.fingerprint = lambda: config.settings.openai_api_key
        router = use_router(bad)
        with self.assertRaises(LLMUnavailable):
            router.complete([], [])
        with patch("app.llm.router.time.time", return_value=10**12):
            with self.assertRaises(LLMUnavailable):
                router.complete([], [])
        self.assertEqual(len(bad.calls), 1)
        config.settings.openai_api_key = "new-key"
        self.assertEqual(router.complete([], []).text, "fixed")

    def test_http_error_redacts_payload_and_parses_quota_retry_after(self):
        response = FakeResponse(429, {"error": {"code": "insufficient_quota", "message": "PRIVATE SECRET"}}, headers={"retry-after": "90"})
        error = http_error(response, "test")
        self.assertEqual(error.category, "billing")
        self.assertEqual(error.retry_after, 90)
        self.assertNotIn("PRIVATE SECRET", str(error))

    def test_openrouter_free_enforces_price_cap_and_rejects_paid_model(self):
        config.settings.openrouter_api_key = "test"
        p = openrouter_provider()
        self.http.route("POST", "openrouter.ai/api/v1/chat/completions", FakeResponse(200, {
            "choices": [{"message": {"content": "free answer"}}]}))
        p.complete([], [], [], "default", 100)
        body = self.http.calls[0]["json"]
        self.assertEqual(body["provider"]["max_price"]["prompt"], 0)
        self.assertTrue(body["provider"]["require_parameters"])
        self.assertEqual(p.cost_class(), "free")
        config.settings.openrouter_model = "paid/model"
        with self.assertRaises(LLMError):
            p.complete([], [], [], "default", 100)
        self.assertEqual(len(self.http.calls), 1)

    def test_gemini_thought_signatures_survive_tool_rounds_only_for_google(self):
        config.settings.gemini_api_key = "test"
        extra = {"google": {"thought_signature": "opaque-signature"}}
        self.http.route("POST", "generativelanguage.googleapis.com", FakeResponse(200, {
            "choices": [{"message": {"tool_calls": [{"id": "g1", "function": {"name": "test", "arguments": "{}"}, "extra_content": extra}]}}]}))
        result = gemini_provider().complete([], [], None, "default", 100)
        self.assertEqual(result.tool_calls[0].extra_content, extra)
        messages = [{"role": "assistant", "tool_calls": result.tool_calls}]
        self.assertEqual(convert_messages([], messages, google=True)[0]["tool_calls"][0]["extra_content"], extra)
        self.assertNotIn("extra_content", convert_messages([], messages)[0]["tool_calls"][0])

    def test_free_tier_is_a_declaration_not_guessed_from_the_api_key(self):
        p = gemini_provider()
        config.settings.gemini_free_tier = False
        self.assertEqual(p.cost_class(), "paid")
        config.settings.gemini_free_tier = True
        self.assertEqual(p.cost_class(), "free")

    def test_malformed_success_payload_also_falls_back(self):
        config.settings.gemini_api_key = "test"
        self.http.route("POST", "generativelanguage.googleapis.com", FakeResponse(200, {"choices": [{"message": {"content": {"unexpected": "private payload"}}}]}))
        fallback = FreeProvider(["fallback"], name="fallback")
        router = use_router(gemini_provider(), fallback)
        self.assertEqual(router.complete([], []).text, "fallback")
        self.assertNotIn("private payload", str(audit.recent()))

    def test_retry_after_http_date_and_invalid_numbers(self):
        from email.utils import formatdate
        with patch("app.llm.errors.time.time", return_value=1000):
            error = http_error(FakeResponse(429, {}, headers={"retry-after": formatdate(1120, usegmt=True)}), "test")
            self.assertEqual(error.retry_after, 120)
        for header in ("Infinity", "NaN", "invalid"):
            self.assertIsNone(http_error(FakeResponse(429, {}, headers={"retry-after": header}), "test").retry_after)
