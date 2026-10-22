from copy import deepcopy
import unittest
from unittest import mock

from research.kira_model import conversation_eval as ev


class FreeConversationTests(unittest.TestCase):
    def test_generated_answers_feed_followup_and_sessions_reset(self):
        scenarios = [{"id": "a", "origin": "test", "turns": [
            {"user": "Premier", "review": "Relire"}, {"user": "Suite", "review": "Relire"}]},
            {"id": "b", "origin": "test", "turns": [{"user": "Autre", "review": "Relire"}]}]
        received = []
        def reply(model, codec, messages, *args, **kwargs):
            received.append(deepcopy(messages))
            return "Sortie " + str(len(received)), messages
        with mock.patch("research.kira_model.dialogue.reply", side_effect=reply):
            report = ev.run_sessions(None, None, scenarios)
        self.assertEqual(received[1], [{"role": "user", "content": "Premier"},
                                      {"role": "assistant", "content": "Sortie 1"},
                                      {"role": "user", "content": "Suite"}])
        self.assertEqual(received[2], [{"role": "user", "content": "Autre"}])
        self.assertFalse(report["general_assistant_qualified"])
        self.assertEqual(report["human_review"], "required-unscored")

    def test_structural_checks_do_not_claim_semantic_grading(self):
        turn = {"required_word": "KIRA"}
        self.assertIn("empty-response", ev.flags_for("", {}))
        self.assertIn("required-word-absent:KIRA", ev.flags_for("nonjour.", turn))
        self.assertIn("replacement-character", ev.flags_for("a\ufffd", {}))
        self.assertIn("repeated-word-triple", ev.flags_for("je suis ici je suis ici je suis ici", {}))
        # Une mention du bon nom peut rester une mauvaise réponse.
        self.assertEqual(ev.flags_for("KIRA donc la bleu.", turn), [])

    def test_dropped_history_is_reported(self):
        cases = [{"id": "a", "origin": "test", "turns": [
            {"user": "A", "review": "Relire"}, {"user": "B", "review": "Relire"}]}]
        with mock.patch("research.kira_model.dialogue.reply", side_effect=lambda m, c, h, *a, **k: ("Réponse", h[-1:])):
            report = ev.run_sessions(None, None, cases)
        self.assertEqual(report["cases"][0]["turns"][1]["discarded_history_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
