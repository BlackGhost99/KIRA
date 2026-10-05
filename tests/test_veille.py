"""Veille : lecture des flux, résumé, déduplication, validation vers la mémoire."""
import json
import unittest

from app import config, consolidate, db, memory, veille
from app.llm.base import LLMError
from tests.helpers import FakeResponse, KiraTestCase, ScriptedProvider, use_router

RSS = """<?xml version="1.0"?><rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel><title>Physique</title>
<item><title>Les ondes gravitationnelles expliquées</title><link>https://exemple.org/ondes</link><pubDate>Mon, 05 Oct 2026 08:00:00 GMT</pubDate>
<description>&lt;p&gt;Une introduction claire aux ondes gravitationnelles : comment deux trous noirs qui fusionnent déforment l'espace-temps et comment LIGO les détecte à des milliards d'années-lumière.&lt;/p&gt;</description></item>
<item><title>Trop court</title><link>https://exemple.org/court</link><description>court</description></item>
<item><title>Le principe de moindre action</title><link>https://exemple.org/action</link>
<content:encoded>&lt;div&gt;La mécanique lagrangienne reformule les lois de Newton à partir d'un principe variationnel : le système suit le chemin qui rend l'action stationnaire, ce qui unifie la mécanique, l'optique et plus tard la théorie des champs.&lt;/div&gt;</content:encoded></item>
</channel></rss>"""

ATOM = """<?xml version="1.0" encoding="utf-8"?><feed xmlns="http://www.w3.org/2005/Atom"><title>Blog</title>
<entry><title>Git pour débutants</title><link rel="self" href="https://b.example/self"/><link rel="alternate" href="https://b.example/git"/>
<updated>2026-10-01T10:00:00Z</updated><summary>Apprendre git pas à pas : initialiser un dépôt, valider des changements, créer des branches, fusionner proprement et résoudre les conflits sans stress, avec des exemples concrets.</summary></entry></feed>"""

JSON_OK = json.dumps({"resume": "Les ondes gravitationnelles sont des vibrations de l'espace-temps.", "sujet": "physique",
                      "niveau": "débutant", "pertinence": 0.8})


class ParsingTests(KiraTestCase):
    def test_rss_with_encoded_content(self):
        entries = veille.parse_feed(RSS.encode())
        self.assertEqual([e["title"] for e in entries], ["Les ondes gravitationnelles expliquées", "Trop court", "Le principe de moindre action"])
        self.assertIn("lagrangienne", entries[2]["summary"])
        self.assertEqual(entries[0]["link"], "https://exemple.org/ondes")

    def test_atom_prefers_alternate_link(self):
        entries = veille.parse_feed(ATOM.encode())
        self.assertEqual(entries[0]["link"], "https://b.example/git")
        self.assertEqual(entries[0]["published"], "2026-10-01T10:00:00Z")

    def test_strip_html_and_invalid_xml(self):
        self.assertEqual(veille.strip_html("<p>Salut <b>toi</b></p>"), "Salut toi")
        with self.assertRaises(Exception):
            veille.parse_feed(b"<html><body>pas un flux")

    def test_add_source_validation(self):
        with self.assertRaises(ValueError):
            veille.add_source("javascript:alert(1)")
        src = veille.add_source("https://exemple.org/feed", "Mon flux", "rss", 7)
        self.assertEqual(src["score"], 1.0)
        updated = veille.update_source(src["id"], active=False, score=0.4)
        self.assertEqual((updated["active"], updated["score"]), (0, 0.4))


class RunTests(KiraTestCase):
    def setUp(self):
        super().setUp()
        db.get_db().run("DELETE FROM veille_sources")
        self.src = veille.add_source("https://exemple.org/feed.xml", "Physique", "rss", 0.9)
        self.http.route("GET", "exemple.org/feed.xml", FakeResponse(200, body=RSS.encode(), headers={"content-type": "application/rss+xml"}))

    def test_seed_defaults_only_once_and_keeps_old_pages_inactive(self):
        db.get_db().run("DELETE FROM veille_sources")
        self.assertEqual(veille.seed_defaults(), len(veille.DEFAULT_SOURCES))
        self.assertEqual(veille.seed_defaults(), 0)
        sources = veille.list_sources()
        self.assertTrue(any(s["active"] for s in sources))
        self.assertFalse(any(s["active"] for s in sources if s["kind"] == "page"))

    def test_run_summarizes_scores_and_stores_pending(self):
        use_router(ScriptedProvider([JSON_OK, JSON_OK]))
        stats = veille.run()
        self.assertEqual((stats["new"], stats["by_llm"], stats["errors"]), (2, 2, []))
        items = veille.list_items("pending")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["score"], 0.72)  # 0.9 (fiabilité) x 0.8 (pertinence)
        self.assertEqual(items[0]["topic"], "physique")
        self.assertEqual(items[0]["source_name"], "Physique")
        self.assertIsNone(veille.list_sources()[0]["last_error"])
        self.assertTrue(db.get_db().kv_get("veille_last_run"))

    def test_second_run_deduplicates(self):
        use_router(ScriptedProvider([JSON_OK] * 4))
        veille.run()
        stats = veille.run()
        self.assertEqual((stats["new"], stats["skipped"]), (0, 2))

    def test_without_ai_falls_back_to_excerpt_and_stops_calling_the_ai(self):
        provider = ScriptedProvider([LLMError("pas de crédit", 402, "x")])
        use_router(provider)
        stats = veille.run()
        self.assertEqual((stats["new"], stats["by_llm"]), (2, 0))
        self.assertEqual(len(provider.calls), 1)
        item = veille.list_items("pending")[0]
        self.assertEqual(item["summarized_by"], "extrait")
        self.assertEqual(item["score"], 0.45)

    def test_invalid_json_from_ai_falls_back(self):
        use_router(ScriptedProvider(["voici mon résumé libre", JSON_OK]))
        veille.run()
        by = sorted(i["summarized_by"] for i in veille.list_items("pending"))
        self.assertEqual(by, ["extrait", "extrait"])  # après un échec, plus d'appel à l'IA

    def test_source_error_is_recorded_and_other_sources_continue(self):
        db.get_db().run("UPDATE veille_sources SET score = 0.5")
        broken = veille.add_source("https://casse.example/feed", "Cassé", "rss", 0.99)
        self.http.route("GET", "casse.example", FakeResponse(500, text="panne"))
        use_router(ScriptedProvider([JSON_OK, JSON_OK]))
        stats = veille.run()
        self.assertEqual(stats["new"], 2)
        self.assertEqual(len(stats["errors"]), 1)
        row = db.get_db().q1("SELECT * FROM veille_sources WHERE id = ?", [broken["id"]])
        self.assertIn("HTTPError", row["last_error"])

    def test_total_cap(self):
        use_router(ScriptedProvider([JSON_OK] * 3))
        stats = veille.run(max_total=1)
        self.assertEqual(stats["new"], 1)

    def test_page_source(self):
        page = veille.add_source("https://cours.example/", "Cours", "page", 0.8)
        html = "<html><head><title>Cours</title></head><body><main>" + ("Une longue leçon sur les vecteurs. " * 10) + "</main></body></html>"
        self.http.route("GET", "cours.example", FakeResponse(200, body=html.encode(), headers={"content-type": "text/html"}))
        veille.update_source(self.src["id"], active=False)
        use_router(ScriptedProvider([JSON_OK]))
        self.assertEqual(veille.run()["new"], 1)

    def test_validation_moves_item_to_memory_as_knowledge(self):
        use_router(ScriptedProvider([JSON_OK, JSON_OK]))
        veille.run()
        item = veille.list_items("pending")[0]
        veille.validate(item["id"])
        veille.validate(item["id"])  # idempotent
        knowledge = memory.list_items("knowledge")
        self.assertEqual(len(knowledge), 1)
        self.assertEqual(knowledge[0]["source"], item["url"])
        self.assertIn(item["title"], knowledge[0]["content"])
        self.assertEqual(veille.counts(), {"validated": 1, "pending": 1})
        veille.reject(veille.list_items("pending")[0]["id"])
        self.assertEqual(veille.counts(), {"validated": 1, "rejected": 1})

    def test_validate_many_by_score_and_auto_validation_setting(self):
        use_router(ScriptedProvider([JSON_OK, json.dumps({"resume": "r", "sujet": "autre", "niveau": "", "pertinence": 0.2})]))
        veille.run()
        self.assertEqual(veille.validate_many(0.5), 1)
        self.assertEqual(veille.counts(), {"validated": 1, "pending": 1})

    def test_auto_validate_config_is_off_by_default(self):
        self.assertEqual(config.Settings().veille_auto_validate_min_score, 0.0)
        use_router(ScriptedProvider([JSON_OK, JSON_OK]))
        config.settings.veille_auto_validate_min_score = 0.7
        stats = veille.run()
        self.assertEqual(stats["auto_validated"], 2)

    def test_validated_knowledge_reaches_the_prompt(self):
        use_router(ScriptedProvider([JSON_OK, JSON_OK]))
        veille.run()
        veille.validate_many(0)
        from app import agent
        dynamic = agent.system_dynamic("parle-moi des ondes gravitationnelles")
        self.assertIn("knowledge", dynamic)
        self.assertIn("source : https://exemple.org/ondes", dynamic)


class ConsolidationTests(KiraTestCase):
    def _chat(self, n):
        d = db.get_db()
        d.run("INSERT INTO conversations (id, title, created_at, updated_at) VALUES ('c','t','x','x')")
        for i in range(n):
            d.insert("messages", {"conversation_id": "c", "role": "user", "content": f"question {i} sur les vecteurs", "meta": "{}", "created_at": db.now_iso()})
            d.insert("messages", {"conversation_id": "c", "role": "assistant", "content": f"réponse {i}", "meta": "{}", "created_at": db.now_iso()})

    def test_skips_when_few_exchanges(self):
        self._chat(1)
        use_router(ScriptedProvider(["ne sera pas appelé"]))
        self.assertIn("skipped", consolidate.consolidate())

    def test_extracts_new_profile_items_once(self):
        self._chat(4)
        answer = json.dumps({"items": [{"kind": "profile", "content": "Brice confond vitesse et accélération"},
                                       {"kind": "autre", "content": "Brice prépare un exposé"}, {"kind": "fact", "content": ""}]})
        use_router(ScriptedProvider([answer, answer]))
        self.assertEqual(consolidate.consolidate()["added"], 2)
        self.assertEqual(consolidate.consolidate()["added"], 0)
        self.assertEqual(memory.counts(), {"profile": 1, "fact": 1})
        self.assertEqual(memory.list_items("fact")[0]["source"], "consolidation")


if __name__ == "__main__":
    unittest.main()
