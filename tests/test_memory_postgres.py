"""Intégration réelle PostgreSQL/pgvector, dans une base CI jetable uniquement."""
import os
import unittest
import uuid
from unittest import mock

from app import config, db, memory
from tests.helpers import KiraTestCase


@unittest.skipUnless(os.environ.get("KIRA_TEST_POSTGRES_URL"), "base PostgreSQL CI non configurée")
class PostgresMemoryTests(KiraTestCase):
    def setUp(self):
        super().setUp()
        import psycopg
        from psycopg import sql
        url = os.environ["KIRA_TEST_POSTGRES_URL"]
        admin = psycopg.connect(url, autocommit=True)
        schema = "kira_test_" + uuid.uuid4().hex
        admin.execute("CREATE EXTENSION IF NOT EXISTS vector")
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        self.addCleanup(admin.close)
        self.addCleanup(lambda: admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))))
        database = db.Database(url)
        original = database._connect
        def connect():
            conn = original()
            conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(schema)))
            return conn
        database._connect = connect
        db.get_db().close()
        db._db = database
        # Simule une installation V2 avant l'initialisation de la V3.
        database.run("CREATE TABLE memories (id BIGSERIAL PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL, "
                     "tags TEXT NOT NULL DEFAULT '', source TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
        database.run("INSERT INTO memories (kind,content,created_at,updated_at) VALUES ('note','souvenir migré','2026','2026')")
        database.init()
        config.settings.embedding_url = "https://example.supabase.co/functions/v1/kira-embedding"
        config.settings.embedding_api_key = "test"
        config.settings.embedding_dimensions = 3
        patcher = mock.patch("app.embeddings.embed", return_value=[1.0, 0.0, 0.0])
        self.embed = patcher.start()
        self.addCleanup(patcher.stop)

    def test_v2_migration_and_idempotence(self):
        db.get_db().init()
        old = memory.get(1)
        self.assertEqual((old["content"], old["status"]), ("souvenir migré", "active"))
        memory.update(1, content="souvenir nouveau")
        self.assertEqual([m["content"] for m in memory.history(1)], ["souvenir migré", "souvenir nouveau"])

    def test_pgvector_semantics_and_french_full_text_fuse(self):
        item = memory.add("profile", "ordinateur professionnel utilisé à l'hôtel")
        with mock.patch("app.embeddings.embed", return_value=[0.0, 1.0, 0.0]):
            memory.add("fact", "recette de bananes")
        self.assertTrue(db.get_db().vector_schema)
        self.assertEqual([r["id"] for r in memory.search("PC du boulot")], [item["id"]])
        self.assertEqual([r["id"] for r in memory.search("ordinateur professionnel")], [item["id"]])
        memory.update(item["id"], status="archived")
        self.assertEqual(memory.search("PC du boulot"), [])

    def test_dimension_changes_and_backfill(self):
        memory.add("fact", "ordinateur professionnel")
        config.settings.embedding_dimensions = 4
        self.embed.return_value = [1.0, 0.0, 0.0, 0.0]
        self.assertEqual(memory.search("PC boulot"), [])
        self.assertEqual(memory.backfill()["indexed"], 2)
        self.assertEqual(memory.semantic_status()["indexed"], 2)

    def test_atomic_revision_and_cascading_erasure(self):
        item = memory.add("project", "projet Django")
        with mock.patch("app.memory._record", side_effect=RuntimeError("indisponible")):
            with self.assertRaises(RuntimeError):
                memory.update(item["id"], content="projet Ninja")
        self.assertEqual(memory.get(item["id"])["content"], "projet Django")
        newer = memory.supersede(item["id"], "projet Ninja")
        self.assertEqual(memory.get(item["id"])["superseded_by"], newer["id"])
        memory.delete(newer["id"])
        self.assertEqual(memory.history(newer["id"]), [])
        self.assertIsNone(db.get_db().q1("SELECT * FROM memory_embeddings WHERE memory_id = ?", [newer["id"]]))
