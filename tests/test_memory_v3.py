"""Mémoire V3 : migrations, continuité, historique et récupération hybride sans réseau."""
import json
import sqlite3
from pathlib import Path
from unittest import mock

import requests

from app import agent, config, db, embeddings, memory
from app.tools import memory_tools
from app.tools.registry import ToolContext
from tests.helpers import KiraTestCase
from tests.test_api import ApiTestCase


class MemoryLifecycleTests(KiraTestCase):
    def test_existing_v2_database_migrates_twice_without_losing_records(self):
        path = str(Path(self._tmp) / "v2.db")
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, "
                         "content TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT '', "
                         "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
            conn.execute("INSERT INTO memories (kind, content, created_at, updated_at) VALUES ('fact','ancien','2026','2026')")
        migrated = db.configure("sqlite:///" + path)
        migrated.init()
        item = memory.get(1)
        self.assertEqual((item["content"], item["version"], item["status"]), ("ancien", 1, "active"))
        memory.update(1, content="nouveau")
        self.assertEqual([r["content"] for r in memory.history(1)], ["ancien", "nouveau"])

    def test_revisions_are_atomic_and_delete_erases_history_and_embeddings(self):
        item = memory.add("project", "Architecture Django", importance=0.8)
        with mock.patch("app.memory._record", side_effect=RuntimeError("échec d'écriture")):
            with self.assertRaises(RuntimeError):
                memory.update(item["id"], content="Architecture Ninja")
        self.assertEqual(memory.get(item["id"])["content"], "Architecture Django")
        self.assertEqual(len(memory.history(item["id"])), 1)
        revised = memory.update(item["id"], content="Architecture Ninja")
        self.assertEqual(revised["version"], 2)
        self.assertEqual([r["content"] for r in memory.history(item["id"])], ["Architecture Django", "Architecture Ninja"])
        memory.delete(item["id"])
        self.assertEqual(memory.history(item["id"]), [])
        self.assertEqual(db.get_db().q("SELECT * FROM memory_embeddings"), [])

    def test_supersede_conserves_old_information_and_excludes_it_from_recall(self):
        item = memory.add("project", "Gaboshop utilise Angular")
        newer = memory.supersede(item["id"], "Gaboshop utilise React")
        self.assertEqual(memory.get(item["id"])["superseded_by"], newer["id"])
        self.assertEqual(memory.search("Angular"), [])
        self.assertEqual(memory.search("Angular", status="superseded")[0]["id"], item["id"])
        self.assertEqual(memory.counts(), {"project": 1})
        with self.assertRaises(ValueError):
            memory.supersede(item["id"], "autre")
        self.assertEqual(len(memory.history(item["id"])), 2)

    def test_archive_keeps_memory_and_restoration_is_explicit(self):
        item = memory.add("episodic", "Nous avons configuré Supabase")
        memory.update(item["id"], status="archived")
        self.assertEqual(memory.search("Supabase"), [])
        self.assertEqual(memory.list_items(status="archived")[0]["id"], item["id"])
        memory.update(item["id"], status="active")
        self.assertEqual(memory.search("Supabase")[0]["id"], item["id"])

    def test_identity_survives_database_reconfiguration_and_enters_prompt(self):
        identity = memory.identity()
        url = db.get_db().url
        db.configure(url)
        self.assertEqual(memory.identity(), identity)
        memory.add("identity", "Je conserve l'histoire de mes évolutions")
        prompt = agent.system_dynamic("Bonjour")
        self.assertIn(identity["id"], prompt)
        self.assertIn("histoire de mes évolutions", prompt)

    def test_self_memory_cannot_be_mutated_by_remember(self):
        result = memory_tools.remember({"kind": "identity", "content": "autre identité"}, ToolContext())
        self.assertIn("propriétaire", result)
        self.assertEqual(memory.counts(), {})

    def test_metadata_validation_and_cognitive_usage(self):
        for value in (-1, 2, float("nan"), "invalide"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                memory.add("goal", "apprendre", confidence=value)
        item = memory.add("goal", "Apprendre les mathématiques")
        memory.search("mathématiques")
        self.assertEqual(memory.get(item["id"])["use_count"], 0)
        memory.mark_used([item, item])
        self.assertEqual(memory.get(item["id"])["use_count"], 1)


class HybridMemoryTests(KiraTestCase):
    def setUp(self):
        super().setUp()
        config.settings.embedding_url = "https://example.supabase.co/functions/v1/kira-embedding"
        config.settings.embedding_api_key = "secret-test"
        config.settings.embedding_dimensions = 3
        patcher = mock.patch("app.embeddings.embed", return_value=[1.0, 0.0, 0.0])
        self.embed = patcher.start()
        self.addCleanup(patcher.stop)

    def test_semantics_finds_synonyms_without_any_matching_words(self):
        item = memory.add("profile", "ordinateur professionnel utilisé à l'hôtel")
        with mock.patch("app.embeddings.embed", return_value=[0.0, 1.0, 0.0]):
            memory.add("fact", "bananes plantains")
        hits = memory.search("PC du boulot")
        self.assertEqual([r["id"] for r in hits], [item["id"]])
        self.assertEqual(memory.search("PC du boulot", kinds=("fact",)), [])

    def test_model_changes_never_compare_incompatible_embeddings(self):
        memory.add("fact", "ordinateur professionnel")
        config.settings.embedding_model = "multilingual-v2"
        self.embed.reset_mock()
        self.assertEqual(memory.search("PC du boulot"), [])
        self.embed.assert_not_called()
        self.assertEqual(memory.backfill()["indexed"], 1)
        self.assertEqual(len(memory.search("PC du boulot")), 1)

    def test_update_invalidates_embedding_and_failure_keeps_lexical_memory(self):
        item = memory.add("fact", "ordinateur professionnel")
        with mock.patch("app.embeddings.embed", side_effect=embeddings.EmbeddingUnavailable("panne")):
            memory.update(item["id"], content="recette de bananes")
            self.assertEqual(memory.search("PC du boulot"), [])
            self.assertEqual(memory.search("bananes")[0]["id"], item["id"])
        self.assertEqual(db.get_db().q("SELECT * FROM memory_embeddings"), [])
        self.assertEqual(memory.backfill()["indexed"], 1)
        self.assertEqual(memory.backfill()["indexed"], 0)

    def test_embedding_failure_does_not_lose_a_new_memory(self):
        with mock.patch("app.embeddings.embed", side_effect=embeddings.EmbeddingUnavailable("panne")):
            item = memory.add("fact", "le serveur Render")
            self.assertEqual(memory.search("Render")[0]["id"], item["id"])
        self.assertEqual(len(memory.history(item["id"])), 1)

    def test_stale_embedding_result_cannot_overwrite_a_new_revision(self):
        item = memory.add("fact", "ancien")
        def changed_in_flight(text):
            with mock.patch("app.embeddings.configured", return_value=False):
                memory.update(item["id"], content="nouveau")
            return [1.0, 0.0, 0.0]
        with mock.patch("app.embeddings.embed", side_effect=changed_in_flight):
            self.assertFalse(memory.index_item(item["id"]))
        self.assertEqual(db.get_db().q("SELECT * FROM memory_embeddings"), [])

    def test_fusion_deduplicates_and_hides_archives_and_deleted_vectors(self):
        first = memory.add("fact", "ordinateur professionnel")
        second = memory.add("fact", "ordinateur secondaire")
        self.assertEqual(len(memory.search("ordinateur")), 2)
        memory.update(second["id"], status="archived")
        self.assertEqual([r["id"] for r in memory.search("ordinateur")], [first["id"]])
        memory.delete(first["id"])
        self.assertEqual(memory.search("ordinateur"), [])
        self.assertEqual(memory.semantic_status()["indexed"], 0)


class EmbeddingClientTests(KiraTestCase):
    def setUp(self):
        super().setUp()
        config.settings.embedding_url = "https://example.supabase.co/functions/v1/kira-embedding"
        config.settings.embedding_api_key = "secret-test"
        config.settings.embedding_dimensions = 3
        config.settings.embedding_model = "gte-small"

    def response(self, payload, status=200):
        session = mock.MagicMock()
        session.__enter__.return_value = session
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status_code = status
        response.iter_content.return_value = [json.dumps(payload).encode()]
        session.post.return_value = response
        return session

    def test_endpoint_contract_and_normalization(self):
        session = self.response({"model": "gte-small", "embedding": [3, 0, 4]})
        with mock.patch("app.embeddings.requests.Session", return_value=session):
            self.assertEqual(embeddings.embed("bonjour"), [0.6, 0, 0.8])
        self.assertFalse(session.post.call_args.kwargs["allow_redirects"])
        self.assertEqual(session.post.call_args.kwargs["json"]["input"], "bonjour")

    def test_openai_compatible_cloud_response(self):
        config.settings.embedding_model = "@cf/baai/bge-m3"
        session = self.response({"model": "@cf/baai/bge-m3", "data": [{"index": 0, "embedding": [3, 0, 4]}]})
        with mock.patch("app.embeddings.requests.Session", return_value=session):
            self.assertEqual(embeddings.embed("le PC du boulot"), [0.6, 0, 0.8])

    def test_redirects_wrong_models_and_malformed_vectors_are_rejected(self):
        cases = [({"model": "autre", "embedding": [1, 0, 0]}, 200),
                 ({"model": "gte-small", "embedding": [1, 0]}, 200),
                 ({"model": "gte-small", "embedding": [0, 0, 0]}, 200),
                 ({"model": "gte-small", "embedding": [True, 0, 0]}, 200),
                 ({"model": "gte-small", "embedding": [float("nan"), 0, 0]}, 200),
                 ({}, 302)]
        for payload, status in cases:
            with self.subTest(payload=payload, status=status):
                with mock.patch("app.embeddings.requests.Session", return_value=self.response(payload, status)):
                    with self.assertRaises(embeddings.EmbeddingUnavailable):
                        embeddings.embed("bonjour")

    def test_timeout_and_insecure_endpoint_do_not_expose_secrets(self):
        session = self.response({})
        session.post.side_effect = requests.Timeout("secret-test bonjour")
        with mock.patch("app.embeddings.requests.Session", return_value=session):
            with self.assertRaises(embeddings.EmbeddingUnavailable) as caught:
                embeddings.embed("bonjour")
        self.assertNotIn("secret-test", str(caught.exception))
        config.settings.embedding_url = "http://example.com"
        with mock.patch("app.embeddings.requests.Session") as network:
            with self.assertRaises(embeddings.EmbeddingUnavailable):
                embeddings.embed("bonjour")
            network.assert_not_called()


class MemoryV3ApiTests(ApiTestCase):
    def test_new_endpoints_are_owner_only(self):
        for method, path in [("GET", "/api/identity"), ("GET", "/api/memory/1/history"), ("POST", "/api/memory/1/supersede")]:
            self.assertEqual(self.api(method, path, token=None).status, 401)

    def test_history_archive_replacement_and_export(self):
        item = self.api("POST", "/api/memory", {"kind": "project", "content": "utilise Angular", "importance": 0.9}).json()
        mid = item["id"]
        revised = self.api("PATCH", f"/api/memory/{mid}", {"content": "utilise React"}).json()
        self.assertEqual(revised["version"], 2)
        self.assertEqual(len(self.api("GET", f"/api/memory/{mid}/history").json()["items"]), 2)
        self.api("PATCH", f"/api/memory/{mid}", {"status": "archived"})
        self.assertEqual(self.api("GET", "/api/memory").json()["items"], [])
        self.assertEqual(self.api("GET", "/api/memory", query="status=archived").json()["counts"], {"project": 1})
        self.api("PATCH", f"/api/memory/{mid}", {"status": "active"})
        replaced = self.api("POST", f"/api/memory/{mid}/supersede", {"content": "utilise Next.js"})
        self.assertEqual(replaced.status, 200)
        export = self.api("GET", "/api/export").json()
        self.assertEqual(len(export["memories"]), 2)
        self.assertEqual(export["identity"], self.api("GET", "/api/identity").json()["identity"])
        self.assertGreater(len(export["memory_history"]), 2)

    def test_invalid_metadata_and_missing_memory(self):
        self.assertEqual(self.api("POST", "/api/memory", {"content": "x", "confidence": 2}).status, 400)
        self.assertEqual(self.api("GET", "/api/memory", query="status=invalide").status, 400)
        self.assertEqual(self.api("GET", "/api/memory/999/history").status, 404)
        self.assertEqual(self.api("POST", "/api/memory/999/supersede", {"content": "x"}).status, 404)
