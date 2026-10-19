"""Même contrats de continuité sur PostgreSQL réel et schéma isolé."""
import os
import unittest
import uuid

from app import auth, db
from tests import test_continuity


class PostgresMixin:
    def setUp(self):
        super().setUp()
        import psycopg
        from psycopg import sql
        url = os.environ['KIRA_TEST_POSTGRES_URL']
        admin = psycopg.connect(url, autocommit=True)
        schema = 'kira_cont_' + uuid.uuid4().hex
        self.admin, self.schema = admin, schema
        admin.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
        self.addCleanup(admin.close)
        self.addCleanup(lambda: admin.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema))))
        database = db.Database(url)
        original = database._connect
        def connect():
            conn = original()
            conn.execute(sql.SQL('SET search_path TO {}, public').format(sql.Identifier(schema)))
            return conn
        database._connect = connect
        db.get_db().close()
        db._db = database
        database.init()
        if hasattr(self, 'token'):
            self.session, self.token, self.refresh = auth.create_session()


def pg(cls):
    return unittest.skipUnless(os.environ.get('KIRA_TEST_POSTGRES_URL'), 'base PostgreSQL CI non configurée')(cls)


@pg
class PostgresEvidence(PostgresMixin, test_continuity.MemoryEvidenceTests):
    pass


@pg
class PostgresJobs(PostgresMixin, test_continuity.DurableJobTests):
    pass


@pg
class PostgresWorkspace(PostgresMixin, test_continuity.WorkspaceTests):
    def test_public_role_cannot_read_projects_documents_jobs_or_files(self):
        from psycopg import sql
        from app import workspace
        role = 'kira_cont_reader_' + uuid.uuid4().hex
        self.admin.execute(sql.SQL('CREATE ROLE {}').format(sql.Identifier(role)))
        def cleanup():
            self.admin.execute(sql.SQL('DROP OWNED BY {}').format(sql.Identifier(role)))
            self.admin.execute(sql.SQL('DROP ROLE {}').format(sql.Identifier(role)))
        self.addCleanup(cleanup)
        self.admin.execute(sql.SQL('GRANT USAGE ON SCHEMA {} TO {}').format(sql.Identifier(self.schema), sql.Identifier(role)))
        self.admin.execute(sql.SQL('GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}').format(sql.Identifier(self.schema), sql.Identifier(role)))
        project = workspace.create_project('Privé')
        workspace.add_task(project['id'], 'Tâche privée')
        workspace.add_document(project['id'], 'secret.txt', b'Document prive')
        test_continuity.wait_all()
        with db.get_db().transaction():
            db.get_db()._conn().execute(sql.SQL('SET LOCAL ROLE {}').format(sql.Identifier(role)))
            for table in ('projects', 'project_tasks', 'documents', 'document_pages', 'learning_cards', 'files', 'background_jobs', 'evaluation_runs'):
                self.assertEqual(db.get_db().q(f'SELECT * FROM {table}'), [])



@pg
class PostgresBackup(PostgresMixin, test_continuity.BackupTests):
    pass


@pg
class PostgresApproval(PostgresMixin, test_continuity.ExactApprovalTests):
    pass


@pg
class PostgresApi(PostgresMixin, test_continuity.ContinuityApiTests):
    pass
