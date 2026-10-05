"""Fournisseurs d'IA, bascule automatique, budget."""
import json
import unittest

import requests

from app import budget, config
from app.llm import BudgetExceeded, LLMUnavailable
from app.llm.anthropic_provider import AnthropicProvider, convert_messages as anth_convert
from app.llm.base import LLMError, LLMResult, ToolCall, ToolSpec
from app.llm.openai_compat import convert_messages as oa_convert, deepseek_provider, ollama_provider, openai_provider
from tests.helpers import FakeResponse, KiraTestCase, ScriptedProvider, use_router

TOOL = ToolSpec("python", "calcule", {"type": "object", "properties": {"code": {"type": "string"}}})
HISTORY = [
    {"role": "user", "content": "Calcule 2+2"},
    {"role": "assistant", "content": "Je calcule.", "tool_calls": [ToolCall("t1", "python", {"code": "print(4)"}),
                                                                    ToolCall("t2", "python", {"code": "print(5)"})]},
    {"role": "tool", "tool_call_id": "t1", "name": "python", "content": "4"},
    {"role": "tool", "tool_call_id": "t2", "name": "python", "content": ""},
]


class ConversionTests(KiraTestCase):
    def test_anthropic_groups_tool_results_in_one_user_message(self):
        out = anth_convert(HISTORY)
        self.assertEqual([m["role"] for m in out], ["user", "assistant", "user"])
        self.assertEqual([b["type"] for b in out[1]["content"]], ["text", "tool_use", "tool_use"])
        results = out[2]["content"]
        self.assertEqual([r["tool_use_id"] for r in results], ["t1", "t2"])
        self.assertEqual(results[1]["content"], "(vide)")

    def test_openai_format(self):
        out = oa_convert(["Système A", "Système B"], HISTORY)
        self.assertEqual(out[0], {"role": "system", "content": "Système A\n\nSystème B"})
        assistant = out[2]
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "python")
        self.assertEqual(json.loads(assistant["tool_calls"][1]["function"]["arguments"]), {"code": "print(5)"})
        self.assertEqual([m["role"] for m in out[3:]], ["tool", "tool"])


class AnthropicTests(KiraTestCase):
    def setUp(self):
        super().setUp()
        config.settings.anthropic_api_key = "sk-ant-test"

    def test_request_shape_and_parsing(self):
        self.http.route("POST", "api.anthropic.com/v1/messages", FakeResponse(200, {
            "content": [{"type": "text", "text": "Voilà."}, {"type": "tool_use", "id": "tu1", "name": "python", "input": {"code": "1+1"}}],
            "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 50}, "stop_reason": "tool_use"}))
        res = AnthropicProvider().complete(["stable", "dynamique", ""], [{"role": "user", "content": "x"}], [TOOL], "deep", 2000)
        sent = self.http.calls[0]
        self.assertEqual(sent["headers"]["x-api-key"], "sk-ant-test")
        self.assertEqual(sent["headers"]["anthropic-version"], "2023-06-01")
        self.assertEqual(sent["json"]["model"], config.settings.anthropic_model_deep)
        self.assertEqual(len(sent["json"]["system"]), 2)  # le bloc vide est écarté
        self.assertIn("cache_control", sent["json"]["system"][0])
        self.assertNotIn("cache_control", sent["json"]["system"][1])
        self.assertEqual(sent["json"]["tools"][0]["input_schema"]["type"], "object")
        self.assertEqual(res.text, "Voilà.")
        self.assertEqual(res.tool_calls[0].arguments, {"code": "1+1"})
        self.assertEqual((res.input_tokens, res.output_tokens), (150, 20))

    def test_tiers_choose_models(self):
        p = AnthropicProvider()
        self.assertEqual(p.model_for("fast"), config.settings.anthropic_model_fast)
        self.assertEqual(p.model_for("default"), config.settings.anthropic_model)

    def test_http_error_raises_llm_error_with_status(self):
        self.http.route("POST", "anthropic.com", FakeResponse(429, text="rate limited"))
        with self.assertRaises(LLMError) as ctx:
            AnthropicProvider().complete([], [{"role": "user", "content": "x"}], None, "default", 100)
        self.assertEqual(ctx.exception.status, 429)


class OpenAICompatTests(KiraTestCase):
    def test_deepseek_tool_call_parsing_and_bad_arguments(self):
        config.settings.deepseek_api_key = "ds-key"
        self.http.route("POST", "api.deepseek.com/chat/completions", FakeResponse(200, {
            "choices": [{"message": {"content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "python", "arguments": "{\"code\": \"print(1)\"}"}},
                {"id": "c2", "function": {"name": "python", "arguments": "pas du json"}}]}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 8}}))
        res = deepseek_provider().complete(["s"], [{"role": "user", "content": "x"}], [TOOL], "default", 500)
        call = self.http.calls[0]
        self.assertEqual(call["headers"]["authorization"], "Bearer ds-key")
        self.assertIn("max_tokens", call["json"])
        self.assertEqual(call["json"]["tools"][0]["function"]["name"], "python")
        self.assertEqual(res.tool_calls[0].arguments, {"code": "print(1)"})
        self.assertEqual(res.tool_calls[1].arguments, {})
        self.assertEqual((res.input_tokens, res.output_tokens), (30, 8))

    def test_openai_uses_max_completion_tokens(self):
        config.settings.openai_api_key = "sk-oa"
        self.http.route("POST", "api.openai.com/v1/chat/completions", FakeResponse(200, {
            "choices": [{"message": {"content": "Salut"}, "finish_reason": "stop"}]}))
        res = openai_provider().complete(["s"], [{"role": "user", "content": "x"}], None, "fast", 300)
        body = self.http.calls[0]["json"]
        self.assertIn("max_completion_tokens", body)
        self.assertEqual(body["model"], config.settings.openai_model_fast)
        self.assertNotIn("tools", body)
        self.assertEqual(res.text, "Salut")

    def test_ollama_needs_no_key_but_a_url(self):
        p = ollama_provider()
        self.assertFalse(p.configured())
        config.settings.ollama_url = "http://localhost:11434/"
        self.assertTrue(p.configured())
        self.http.route("POST", "localhost:11434/v1/chat/completions", FakeResponse(200, {
            "choices": [{"message": {"content": "ok"}}]}))
        self.assertEqual(p.complete([], [{"role": "user", "content": "x"}], None, "default", 50).text, "ok")
        self.assertNotIn("authorization", self.http.calls[0]["headers"])

    def test_unexpected_payload_is_an_llm_error(self):
        config.settings.groq_api_key = "k"
        from app.llm.openai_compat import groq_provider
        self.http.route("POST", "groq.com", FakeResponse(200, {"oups": 1}))
        with self.assertRaises(LLMError):
            groq_provider().complete([], [{"role": "user", "content": "x"}], None, "default", 50)


class RouterTests(KiraTestCase):
    def test_failover_to_next_provider_and_audit(self):
        bad = ScriptedProvider([LLMError("HTTP 529 surchargé", 529, "a")], name="a")
        good = ScriptedProvider(["Réponse B"], name="b")
        router = use_router(bad, good)
        res = router.complete(["s"], [{"role": "user", "content": "x"}])
        self.assertEqual((res.text, res.provider), ("Réponse B", "b"))
        from app import audit
        self.assertIn("provider_failover", {e["action"] for e in audit.recent()})
        status = {s["name"]: s for s in router.status()}
        self.assertGreater(status["a"]["cooldown_s"], 0)

    def test_provider_in_cooldown_is_skipped_next_time(self):
        a = ScriptedProvider([requests.ConnectionError("réseau coupé"), "ne devrait pas servir"], name="a")
        b = ScriptedProvider(["B1", "B2"], name="b")
        router = use_router(a, b)
        self.assertEqual(router.complete([], [{"role": "user", "content": "1"}]).text, "B1")
        self.assertEqual(router.complete([], [{"role": "user", "content": "2"}]).text, "B2")
        self.assertEqual(len(a.calls), 1)

    def test_all_failing_raises_unavailable_but_retries_resting_ones(self):
        a = ScriptedProvider([LLMError("panne", 500, "a"), "revenu"], name="a")
        router = use_router(a)
        with self.assertRaises(LLMUnavailable):
            router.complete([], [{"role": "user", "content": "x"}])
        self.assertEqual(router.complete([], [{"role": "user", "content": "x"}]).text, "revenu")

    def test_priority_setting_orders_providers(self):
        a = ScriptedProvider(["A"], name="anthropic")
        d = ScriptedProvider(["D"], name="deepseek")
        router = use_router(a, d)
        config.settings.llm_priority = "deepseek,anthropic"
        self.assertEqual(router.complete([], [{"role": "user", "content": "x"}]).text, "D")

    def test_no_provider_configured_message(self):
        from app.llm import Router
        router = Router()  # vrais fournisseurs, aucune clé
        with self.assertRaises(LLMUnavailable) as ctx:
            router.complete([], [{"role": "user", "content": "x"}])
        self.assertIn("ANTHROPIC_API_KEY", str(ctx.exception))

    def test_usage_is_recorded_and_budget_blocks(self):
        config.settings.daily_token_budget = 40
        router = use_router(ScriptedProvider(["a", "b", "c", "d"], name="a"))
        for _ in range(2):  # 15 jetons par appel
            router.complete([], [{"role": "user", "content": "x"}], purpose="test")
        self.assertEqual(budget.used_today(), 30)
        router.complete([], [{"role": "user", "content": "x"}])
        with self.assertRaises(BudgetExceeded):
            router.complete([], [{"role": "user", "content": "x"}])
        summary = budget.summary()
        self.assertEqual(summary["by_provider"], {"a": 45})

    def test_budget_zero_means_unlimited(self):
        config.settings.daily_token_budget = 0
        router = use_router(ScriptedProvider(["a"] * 5, name="a"))
        for _ in range(5):
            router.complete([], [{"role": "user", "content": "x"}])


if __name__ == "__main__":
    unittest.main()
