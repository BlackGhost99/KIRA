"""Une analyse volumineuse ne doit pas rendre le fournisseur indisponible pour le chat."""
import json

from app import config, evolution
from app.llm.base import LLMError, LLMUnavailable
from app.llm.errors import http_error
from tests.helpers import FakeResponse, KiraTestCase, ScriptedProvider, use_router


class FreeGroq(ScriptedProvider):
    def __init__(self, script):
        super().__init__(script, name='groq')

    def cost_class(self):
        return 'free'


class RequestCapacityTests(KiraTestCase):
    def test_413_is_redacted_and_does_not_block_a_later_small_chat(self):
        error = http_error(FakeResponse(413, {'error': {'message': 'PRIVATE CONTENT'}}), 'groq')
        self.assertEqual(error.category, 'request_too_large')
        self.assertNotIn('PRIVATE CONTENT', str(error))
        provider = FreeGroq([error, 'petite réponse'])
        router = use_router(provider)
        with self.assertRaises(LLMUnavailable) as caught:
            router.complete([], [{'role': 'user', 'content': 'large'}], purpose='evolution')
        self.assertIn('trop volumineuse', str(caught.exception))
        self.assertTrue(router.status()[0]['ready'])
        self.assertEqual(router.status()[0]['cooldown_s'], 0)
        self.assertEqual(router.complete([], [{'role': 'user', 'content': 'bonjour'}]).text, 'petite réponse')

    def test_413_can_fail_over_without_disabling_the_first_provider(self):
        first = FreeGroq([LLMError('large', 413)])
        fallback = ScriptedProvider(['réponse de secours'], name='fallback')
        router = use_router(first, fallback)
        self.assertEqual(router.complete([], []).provider, 'fallback')
        self.assertTrue(next(p for p in router.status() if p['name'] == 'groq')['ready'])

    def test_real_cooldown_has_an_actionable_message(self):
        provider = FreeGroq([LLMError('quota', 429, retry_after=900)])
        router = use_router(provider)
        with self.assertRaises(LLMUnavailable):
            router.complete([], [])
        with self.assertRaisesRegex(LLMUnavailable, 'temporairement en pause'):
            router.complete([], [])
        self.assertEqual(len(provider.calls), 1)


class CompactEvolutionTests(KiraTestCase):
    def assert_bounded(self, call):
        size = sum(len(s.encode('utf-8')) for s in call['system'])
        size += sum(len(m['content'].encode('utf-8')) for m in call['messages'])
        self.assertLessEqual(size, evolution.COMPACT_INPUT_BYTES)
        self.assertEqual(call['max_tokens'], evolution.COMPACT_OUTPUT_TOKENS)

    def test_snapshot_budget_includes_omission_notices_and_separators(self):
        for limit in (0, 1, 64, 1000, 8000):
            with self.subTest(limit=limit):
                self.assertLessEqual(len(evolution.source_snapshot(limit)), limit)
                self.assertLessEqual(len(evolution.source_snapshot(limit, byte_budget=True).encode('utf-8')), limit)

    def test_free_analysis_and_correction_stay_bounded_with_unicode(self):
        config.settings.llm_routing_mode = 'ECONOMY'
        provider = FreeGroq(['réponse invalide ' * 2000, json.dumps({'proposal': None, 'reason': 'Périmètre trop limité.'})])
        use_router(provider)
        self.assertEqual(evolution.propose('Écran et boutons : ' * 3000)['status'], 'none')
        self.assertEqual(len(provider.calls), 2)
        for call in provider.calls:
            self.assert_bounded(call)
        self.assertIn('fichiers entiers', provider.calls[0]['messages'][0]['content'])

    def test_large_analysis_retries_once_with_real_reduced_context(self):
        provider = ScriptedProvider([LLMError('large', 413), json.dumps({'proposal': None})])
        router = use_router(provider)
        self.assertEqual(evolution.propose()['status'], 'none')
        self.assertEqual(len(provider.calls), 2)
        self.assertGreater(len(provider.calls[0]['messages'][0]['content']), len(provider.calls[1]['messages'][0]['content']))
        self.assert_bounded(provider.calls[1])
        self.assertTrue(router.status()[0]['ready'])

    def test_free_groq_also_starts_compact_in_quality_mode(self):
        config.settings.groq_free_tier = True
        provider = FreeGroq([json.dumps({'proposal': None})])
        use_router(provider)
        self.assertEqual(evolution.propose()['status'], 'none')
        self.assertEqual(len(provider.calls), 1)
        self.assert_bounded(provider.calls[0])

    def test_second_413_stops_after_the_single_compact_retry(self):
        provider = ScriptedProvider([LLMError('large', 413), LLMError('still large', 413), 'chat'])
        router = use_router(provider)
        with self.assertRaisesRegex(evolution.EvolutionError, 'analyse compacte'):
            evolution.propose()
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(router.complete([], []).text, 'chat')

    def test_still_oversized_analysis_stops_without_blocking_chat(self):
        config.settings.llm_routing_mode = 'ECONOMY'
        provider = FreeGroq([LLMError('large', 413), 'bonjour'])
        router = use_router(provider)
        with self.assertRaisesRegex(evolution.EvolutionError, 'analyse compacte'):
            evolution.propose()
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(router.complete([], []).text, 'bonjour')

    def test_compact_analysis_still_refuses_protected_changes(self):
        config.settings.llm_routing_mode = 'ECONOMY'
        bad = json.dumps({'proposal': {'title': 'interdit', 'changes': [
            {'path': 'app/principles.py', 'action': 'edit', 'find': 'a', 'replace': 'b'}]}})
        provider = FreeGroq([bad] * 3)
        use_router(provider)
        with self.assertRaisesRegex(evolution.EvolutionError, 'protégé'):
            evolution.propose()
        for call in provider.calls:
            self.assert_bounded(call)
