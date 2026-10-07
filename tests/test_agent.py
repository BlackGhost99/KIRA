"""Boucle de l'agent : outils, mémoire, historique, erreurs."""
import unittest

from app import agent, db, memory, tools
from app.llm.base import LLMError
from tests.helpers import KiraTestCase, ScriptedProvider, text_result, tool_result, use_router


def run(conv, text, tier="default"):
    return list(agent.run_turn(conv, text, tier))


class AgentTests(KiraTestCase):
    def test_simple_answer_is_saved_with_metadata(self):
        provider = ScriptedProvider(["Bonjour Brice !"])
        use_router(provider)
        events = run(None, "Salut KIRA")
        kinds = [e["type"] for e in events]
        self.assertEqual(kinds, ["conversation", "user_saved", "message"])
        msg = events[-1]
        self.assertEqual(msg["content"], "Bonjour Brice !")
        self.assertEqual(msg["provider"], "fake")
        rows = db.get_db().q("SELECT role, content FROM messages ORDER BY id")
        self.assertEqual([(r["role"], r["content"]) for r in rows], [("user", "Salut KIRA"), ("assistant", "Bonjour Brice !")])
        conv = db.get_db().q1("SELECT * FROM conversations")
        self.assertEqual(conv["title"], "Salut KIRA")

    def test_tool_loop_runs_python_and_feeds_result_back(self):
        provider = ScriptedProvider([
            tool_result("python", {"code": "print(6 * 7)"}),
            text_result("Le résultat est 42."),
        ])
        use_router(provider)
        events = run(None, "Combien font 6 fois 7 ?")
        self.assertIn("status", [e["type"] for e in events])
        self.assertEqual(events[-1]["content"], "Le résultat est 42.")
        self.assertEqual(events[-1]["tools"], ["python"])
        second_call_messages = provider.calls[1]["messages"]
        tool_msg = second_call_messages[-1]
        self.assertEqual(tool_msg["role"], "tool")
        self.assertIn("42", tool_msg["content"])
        self.assertTrue(provider.calls[0]["tools"])  # les outils sont proposés

    def test_remember_tool_writes_profile_memory_and_next_turn_sees_it(self):
        provider = ScriptedProvider([
            tool_result("remember", {"content": "Brice veut maîtriser la mécanique classique", "kind": "profile"}),
            text_result("Noté."),
            text_result("Reprenons la mécanique."),
        ])
        use_router(provider)
        first = run(None, "Je veux maîtriser la mécanique classique")
        self.assertEqual(memory.profile()[0]["content"], "Brice veut maîtriser la mécanique classique")
        run(first[0]["id"], "On continue ?")
        dynamic = provider.calls[2]["system"][1]
        self.assertIn("Brice veut maîtriser la mécanique classique", dynamic)
        stable = provider.calls[2]["system"][0]
        self.assertIn("Souveraineté du propriétaire", stable)

    def test_history_is_replayed_but_tool_turns_are_not(self):
        provider = ScriptedProvider([tool_result("recall", {"query": "x"}), text_result("R1"), text_result("R2")])
        use_router(provider)
        events = run(None, "Q1")
        run(events[0]["id"], "Q2")
        replayed = provider.calls[2]["messages"]
        self.assertEqual([(m["role"], m["content"]) for m in replayed], [("user", "Q1"), ("assistant", "R1"), ("user", "Q2")])

    def test_unknown_tool_and_tool_exception_do_not_crash(self):
        provider = ScriptedProvider([tool_result("nexiste_pas", {}), text_result("Ok malgré tout.")])
        use_router(provider)
        events = run(None, "test")
        self.assertEqual(events[-1]["content"], "Ok malgré tout.")
        self.assertIn("outil inconnu", provider.calls[1]["messages"][-1]["content"])

    def test_tool_round_limit_forces_a_final_answer_without_tools(self):
        from app import config
        config.settings.max_tool_rounds = 2
        provider = ScriptedProvider([tool_result("recall", {"query": "a"}, "c1"), tool_result("recall", {"query": "b"}, "c2"),
                                     text_result("Réponse finale après limite.")])
        use_router(provider)
        events = run(None, "boucle")
        self.assertEqual(events[-1]["content"], "Réponse finale après limite.")
        self.assertEqual(provider.calls[2]["tools"], [])

    def test_all_providers_down_yields_error_event_and_keeps_user_message(self):
        use_router(ScriptedProvider([LLMError("panne totale", 500, "x")]))
        events = run(None, "Y a quelqu'un ?")
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("HTTP 500", events[-1]["message"])
        self.assertNotIn("panne totale", events[-1]["message"])
        self.assertEqual(db.get_db().q1("SELECT COUNT(*) AS n FROM messages")["n"], 1)

    def test_empty_message_and_deep_tier(self):
        use_router(ScriptedProvider(["ok"]))
        self.assertEqual(run(None, "   ")[0]["type"], "error")
        provider = ScriptedProvider(["profond"])
        use_router(provider)
        run(None, "question difficile", tier="deep")
        self.assertEqual(provider.calls[0]["tier"], "deep")
        self.assertEqual(provider.calls[0]["max_tokens"], 8192)

    def test_unknown_conversation_id_starts_a_new_conversation(self):
        use_router(ScriptedProvider(["ok"]))
        events = run("n-existe-pas", "bonjour")
        self.assertNotEqual(events[0]["id"], "n-existe-pas")

    def test_system_prompt_contains_owner_date_and_instructions(self):
        provider = ScriptedProvider(["ok"])
        use_router(provider)
        run(None, "bonjour")
        stable, dynamic = provider.calls[0]["system"]
        self.assertIn("Brice", stable)
        self.assertIn("LaTeX", stable)
        self.assertIn("jamais des ordres", stable)
        self.assertRegex(dynamic, r"Date et heure : \w+ \d+ \w+ 20\d\d")


class ToolRegistryTests(KiraTestCase):
    def test_all_tools_registered_with_valid_schemas(self):
        names = {s.name for s in tools.specs()}
        self.assertEqual(names, {"python", "remember", "recall", "web_search", "fetch_url", "devices", "device_action", "device_result"})
        for spec in tools.specs():
            self.assertEqual(spec.parameters["type"], "object")
            self.assertTrue(spec.description)

    def test_long_tool_output_is_truncated(self):
        ctx = tools.ToolContext()
        out = tools.run("python", {"code": "print('x' * 50000)"}, ctx)
        self.assertLess(len(out), 12000)

    def test_recall_finds_memory_and_old_messages(self):
        memory.add("fact", "Le LHC est un accélérateur de particules")
        out = tools.run("recall", {"query": "accélérateur"}, tools.ToolContext())
        self.assertIn("LHC", out)
        self.assertEqual(tools.run("recall", {"query": "zzzzz"}, tools.ToolContext()), "Rien trouvé.")


if __name__ == "__main__":
    unittest.main()
