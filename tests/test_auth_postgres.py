"""Les contrats de session tournent aussi sur une vraie base PostgreSQL CI."""
import os
import unittest
import uuid

from app import auth, budget, db
from tests import test_auth_sessions


@unittest.skipUnless(os.environ.get("KIRA_TEST_POSTGRES_URL"), "base PostgreSQL CI non configurée")
class PostgresSessionTests(test_auth_sessions.SessionTests):
    def setUp(self):
        super().setUp()
        import psycopg
        from psycopg import sql
        url = os.environ["KIRA_TEST_POSTGRES_URL"]
        admin = psycopg.connect(url, autocommit=True)
        self.schema = "kira_auth_" + uuid.uuid4().hex
        self.admin = admin
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema)))
        self.addCleanup(admin.close)
        self.addCleanup(lambda: admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema))))
        database = db.Database(url)
        original = database._connect
        def connect():
            conn = original()
            conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.schema)))
            return conn
        database._connect = connect
        db.get_db().close()
        db._db = database
        database.init()

    def test_public_role_cannot_read_session_hashes(self):
        from psycopg import sql
        role = "kira_auth_reader_" + uuid.uuid4().hex
        self.admin.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(role)))
        def cleanup():
            self.admin.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            self.admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
        self.addCleanup(cleanup)
        self.admin.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(self.schema), sql.Identifier(role)))
        self.admin.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(sql.Identifier(self.schema), sql.Identifier(role)))
        _, _, refresh = auth.create_session()
        auth.refresh_session(refresh)
        with db.get_db().transaction():
            db.get_db()._conn().execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role)))
            self.assertEqual(db.get_db().q("SELECT * FROM owner_sessions"), [])
            self.assertEqual(db.get_db().q("SELECT * FROM owner_refresh_used"), [])

    def test_usage_migration_is_additive_and_idempotent(self):
        database = db.get_db()
        database.run("ALTER TABLE llm_usage DROP COLUMN cost_class")
        database.init()
        database.init()
        budget.record("paid", "m", 12, 3)
        budget.record("free", "m", 12, 3, cost_class="free")
        self.assertEqual(budget.summary()["paid_used"], 15)
        self.assertEqual(budget.used_today(), 30)
