"""Contrats de continuité : mémoire, confidentialité, tâches, accords, projets et restauration."""
import base64
import gzip
import io
import json
from pathlib import Path
import shutil
import threading
import time
import unittest
from unittest.mock import patch

from app import agent, backup, db, evaluation, evolution, jobs, memory, privacy, workspace
from app.approvals import proposal_digest
from app.llm.base import LLMError
from app.main import app
from app.tools.registry import REGISTRY, Tool, ToolSpec
from tests.helpers import KiraTestCase, ScriptedProvider, call, tool_result, use_router
from tests.test_api import ApiTestCase


def wait_all():
    end = time.monotonic() + 20
    while jobs.status()['running'] and time.monotonic() < end:
        time.sleep(.02)
    if jobs.status()['running']:
        raise AssertionError('Traitement non terminé dans le délai du test.')


class MemoryEvidenceTests(KiraTestCase):
    def test_source_and_confirmation_survive_revisions(self):
        p = ScriptedProvider([tool_result('remember', {'kind':'profile','content':'Je préfère les exemples concrets'}), 'Bien retenu.'])
        use_router(p)
        events = list(agent.run_turn(None, 'Retiens cette préférence.'))
        item = memory.profile()[0]
        self.assertEqual(item['evidence_status'], 'inferred')
        self.assertEqual(item['origin_conversation'], events[0]['id'])
        self.assertIsNotNone(db.get_db().q1('SELECT * FROM messages WHERE id = ?', [item['origin_message']]))
        memory.update(item['id'], evidence_status='confirmed')
        self.assertEqual(memory.history(item['id'])[0]['evidence_status'], 'inferred')
        self.assertEqual(memory.get(item['id'])['evidence_status'], 'confirmed')

    def test_private_memories_and_exchanges_stay_outside_public_recall(self):
        with privacy.scope(True):
            item = memory.add('profile', 'Dossier confidentiel Orion', source='chat')
        with patch('app.embeddings.embed') as embed:
            self.assertFalse(memory.index_item(item['id']))
            self.assertEqual(memory.search('confidentiel'), [])
            with privacy.scope(True):
                self.assertEqual(memory.search('confidentiel')[0]['id'], item['id'])
            embed.assert_not_called()
        self.assertEqual(memory.profile(), [])

    def test_private_result_cannot_fail_over_to_unapproved_provider_or_leak_next_turn(self):
        def private_tool(args, ctx):
            ctx.private_data = True
            return 'Résultat confidentiel Orion'
        name = 'unit_private_result'
        REGISTRY[name] = Tool(ToolSpec(name, 'test', {'type':'object'}), private_tool, 'test')
        self.addCleanup(REGISTRY.pop, name, None)
        approved = ScriptedProvider([tool_result(name, {}), LLMError('down', 500)], name='fake')
        unapproved = ScriptedProvider(['must not be called'], name='other')
        use_router(approved, unapproved)
        with patch('app.embeddings.embed') as embed:
            events = list(agent.run_turn(None, 'Lis mon document'))
            cid = events[0]['id']
            self.assertEqual(events[-1]['type'], 'error')
            self.assertEqual(db.get_db().q1('SELECT private FROM conversations WHERE id = ?', [cid])['private'], 1)
            self.assertEqual(unapproved.calls, [])
            use_router(unapproved)
            again = list(agent.run_turn(cid, 'Peux-tu reprendre le confidentiel ?'))
            self.assertEqual(again[-1]['type'], 'error')
            self.assertEqual(unapproved.calls, [])
            embed.assert_not_called()
        self.assertEqual(memory.search_messages('confidentiel'), [])


class DurableJobTests(KiraTestCase):
    def setUp(self):
        super().setUp()
        self.addCleanup(jobs._handlers.pop, 'unit-safe', None)

    def insert_job(self, kind='unit-safe', state='queued'):
        now = db.now_iso()
        db.get_db().insert('background_jobs', {'id':'unit-job','name':'unit','kind':kind,'state':state,'payload':'{}','created_at':now,'updated_at':now})

    def test_result_survives_database_reopen_and_expired_work_is_recovered(self):
        jobs.register('unit-safe', lambda payload,ctx: {'value':42})
        self.insert_job(state='running')
        def reopen():
            current = db.get_db()
            factory = current._connect
            current.close()
            replacement = db.Database(current.url)
            replacement._connect = factory  # conserve aussi le schéma isolé de la connexion PostgreSQL
            db._db = replacement
            replacement.init()
        reopen()
        jobs.recover()
        wait_all()
        self.assertEqual(jobs.status()['last']['unit']['result'], {'value':42})
        reopen()
        self.assertEqual(jobs.status()['items'][0]['state'], 'done')

    def test_two_workers_claim_a_job_once(self):
        seen, entered, release = [], threading.Event(), threading.Event()
        def handler(payload,ctx):
            seen.append(1); entered.set(); release.wait(3); return {'ok':True}
        jobs.register('unit-safe', handler)
        self.insert_job()
        a = threading.Thread(target=jobs._execute, args=('unit-job',))
        b = threading.Thread(target=jobs._execute, args=('unit-job',))
        a.start(); self.assertTrue(entered.wait(3)); b.start(); b.join(3); release.set(); a.join(3)
        self.assertEqual(seen, [1])
        self.assertEqual(jobs.status()['items'][0]['state'], 'done')

    def test_cancellation_is_observed_before_the_next_stage(self):
        entered, release, seen = threading.Event(), threading.Event(), []
        def handler(payload,ctx):
            entered.set(); release.wait(3); ctx.checkpoint(50); seen.append('effect'); return {}
        jobs.register('unit-safe', handler)
        self.assertTrue(jobs.start('unit', kind='unit-safe'))
        self.assertTrue(entered.wait(3))
        jid = jobs.status()['items'][0]['id']
        self.assertTrue(jobs.cancel(jid)); release.set(); wait_all()
        self.assertEqual(seen, [])
        self.assertEqual(jobs.status()['items'][0]['state'], 'cancelled')

    def test_non_idempotent_evolution_waits_for_manual_resume(self):
        self.insert_job(kind='evolution', state='running')
        jobs.recover()
        self.assertEqual(jobs.status()['items'][0]['state'], 'interrupted')


class WorkspaceTests(KiraTestCase):
    def test_simultaneous_edits_are_detected_and_completed_tasks_have_a_result(self):
        p = workspace.create_project('Orion', private=False)
        updated = workspace.patch_project(p['id'], {'goal':'Livrer la première version'}, p['version'])
        self.assertEqual(updated['version'], 2)
        with self.assertRaises(workspace.Conflict):
            workspace.patch_project(p['id'], {'goal':'Écraser la nouvelle version'}, p['version'])
        t = workspace.add_task(p['id'], 'Tester la mémoire')
        with self.assertRaises(ValueError):
            workspace.patch_task(t['id'], 'done', '', t['version'])
        self.assertEqual(workspace.patch_task(t['id'], 'done', 'Test réalisé et vérifié', t['version'])['state'], 'done')

    def test_text_import_has_citable_sections_and_resumable_deduplication(self):
        p = workspace.create_project('Documents')
        doc = workspace.add_document(p['id'], 'notes.txt', 'Le budget du projet Orion est de 12345 unités.'.encode())
        wait_all()
        self.assertEqual(jobs.status()['items'][0]['state'], 'done')
        page = workspace.excerpts(p['id'], 'budget Orion')[0]
        self.assertEqual((page['document_id'],page['location'],page['number']), (doc['id'],'section',1))
        self.assertIn('12345', workspace.prompt(p['id'], 'budget'))
        same = workspace.add_document(p['id'], 'notes.txt', 'Le budget du projet Orion est de 12345 unités.'.encode())
        self.assertEqual(same['id'], doc['id'])
        self.assertEqual(len(workspace.detail(p['id'])['documents']), 1)

    def test_private_project_uses_only_approved_provider_and_reopens_with_its_context(self):
        p = workspace.create_project('Orion', 'Construire une station météo')
        approved, other = ScriptedProvider(['Bonjour','Suite']), ScriptedProvider(['must not run'], name='other')
        use_router(approved, other)
        events = list(agent.run_turn(None, 'Où en sommes-nous ?', project_id=p['id']))
        cid = events[0]['id']
        self.assertIn('station météo', '\n'.join(approved.calls[0]['system']))
        self.assertTrue(events[-1]['private'])
        list(agent.run_turn(cid, 'On continue'))
        self.assertIn('station météo', '\n'.join(approved.calls[1]['system']))
        self.assertEqual(other.calls, [])

    def test_learning_progress_is_durable_and_duplicate_reviews_are_rejected(self):
        p = workspace.create_project('Physique')
        card = workspace.add_card(p['id'], 'Énergie cinétique ?', 'E = m v² / 2', 'Cours, section 2')
        reviewed = workspace.review_card(card['id'], True, card['version'])
        self.assertEqual((reviewed['attempts'],reviewed['correct'],reviewed['interval_days']), (1,1,1))
        with self.assertRaises(workspace.Conflict):
            workspace.review_card(card['id'], True, card['version'])
        self.assertIn('Énergie cinétique', workspace.prompt(p['id'], ''))

    @unittest.skipUnless(shutil.which('pdftoppm') and shutil.which('tesseract'), 'OCR système absent')
    def test_scanned_pdf_ocr_recovers_text_without_network(self):
        from PIL import Image, ImageDraw, ImageFont
        from app.document_parser import extract
        image = Image.new('RGB', (1400,300), 'white')
        ImageDraw.Draw(image).text((60,80), 'Projet Orion Budget 12345', font=ImageFont.load_default(size=50), fill='black')
        raw = io.BytesIO(); image.save(raw, format='PDF')
        pages = extract(raw.getvalue(), '.pdf')
        self.assertEqual(pages[0]['location'], 'page')
        self.assertIn('Orion', pages[0]['content'])
        self.assertIn('12345', pages[0]['content'])


class BackupTests(KiraTestCase):
    def test_round_trip_keeps_identity_history_files_projects_and_learning(self):
        identity = memory.identity()
        p = workspace.create_project('Orion')
        workspace.add_task(p['id'], 'Vérifier la restauration')
        workspace.add_card(p['id'], 'Question', 'Réponse')
        doc = workspace.add_document(p['id'], 'source.txt', b'Preuve documentaire Orion')
        wait_all()
        m = memory.add('fact', 'Version ancienne')
        newer = memory.supersede(m['id'], 'Version nouvelle')
        blob = backup.snapshot()
        self.assertNotIn(b'owner_sessions', gzip.decompress(blob))
        target = db.Database('sqlite:///' + self._tmp + '/restored.db'); target.init()
        self.addCleanup(target.close)
        backup.restore(blob, target)
        self.assertEqual(json.loads(target.kv_get('kira_identity')), identity)
        self.assertEqual(target.q1('SELECT superseded_by FROM memories WHERE id = ?', [m['id']])['superseded_by'], newer['id'])
        self.assertEqual(target.q1('SELECT data FROM files WHERE id = ?', [doc['file_id']])['data'], b'Preuve documentaire Orion')
        self.assertEqual(target.q1('SELECT COUNT(*) AS n FROM learning_cards')['n'], 1)
        self.assertEqual(target.q1('SELECT state FROM background_jobs')['state'], 'interrupted')
        with self.assertRaises(ValueError):
            backup.restore(blob, target)

    def test_corrupted_or_injected_backup_is_rejected_atomically(self):
        p = workspace.create_project('Orion')
        blob = backup.snapshot(); data = backup.unpack(blob)
        data['tables']['projects']['rows'][0]['title'] = 'Corrompu'
        with self.assertRaises(ValueError):
            backup.unpack(gzip.compress(json.dumps(data).encode()))
        data = backup.unpack(blob)
        data['tables']['memories']['rows'].append({'id':1,'untrusted_column; DROP TABLE projects; --':'bad'})
        data['tables']['memories']['sha256'] = backup._checksum(data['tables']['memories']['rows'])
        target = db.Database('sqlite:///' + self._tmp + '/bad.db'); target.init(); self.addCleanup(target.close)
        with self.assertRaises(ValueError):
            backup.restore(gzip.compress(json.dumps(data).encode()), target)
        self.assertEqual(target.q1('SELECT COUNT(*) AS n FROM projects')['n'], 0)


class ExactApprovalTests(KiraTestCase):
    def test_untested_or_changed_proposal_cannot_be_approved(self):
        from tests.test_evolution import GOOD, as_reply
        use_router(ScriptedProvider([as_reply(GOOD)]))
        with patch('app.evolution.run_tests', return_value=(True,'ok')):
            pid = evolution.propose()['id']
        row = evolution.get_proposal(pid)
        db.get_db().update('proposals',pid,{'rationale':'Texte modifié après affichage'})
        with self.assertRaises(evolution.EvolutionError):
            evolution.approve(pid,row['approval_hash'])
        self.assertEqual(self.http.calls, [])
        db.get_db().update('proposals',pid,{'tests_ok':None})
        with self.assertRaises(evolution.EvolutionError):
            evolution.approve(pid)
        self.assertEqual(self.http.calls, [])

    def test_machine_action_changed_after_display_is_not_queued(self):
        from app import devices
        from tests.test_devices import pair_device
        paired = pair_device()
        action = devices.request_action(paired['device_id'], 'write_file', {'path':'/home/b/KIRA/note.txt','content':'Texte présenté'})
        self.assertEqual(action['status'], 'pending')
        token = action['approval_hash']
        changed = {'path':'/home/b/KIRA/note.txt','content':'Texte différent'}
        db.get_db().update('device_actions',action['id'],{'args':db.jdump(changed)})
        with self.assertRaises(devices.DeviceError):
            devices.approve(action['id'],token)
        self.assertEqual(devices.get_action(action['id'])['status'], 'pending')
        reviewed = devices.get_action(action['id'])
        self.assertNotEqual(reviewed['approval_hash'], token)
        self.assertEqual(devices.approve(action['id'],reviewed['approval_hash'])['status'], 'queued')

    def test_evaluation_records_five_synthetic_scenarios_without_executing_tools(self):
        use_router(ScriptedProvider(['42','{"nom":"KIRA","langue":"fr"}','Orion','DOCUMENT',tool_result('lookup',{'id':'alpha'})]))
        class Context:
            def checkpoint(self,progress): pass
        report = evaluation.run({}, Context())
        self.assertEqual((report['passed'],report['total']), (5,5))
        self.assertEqual(len(evaluation.recent()), 1)
        self.assertEqual(db.get_db().q1('SELECT COUNT(*) AS n FROM device_actions')['n'], 0)


class ContinuityApiTests(ApiTestCase):
    def test_new_features_require_owner_access_and_project_conflicts_return_409(self):
        for method,path in [('GET','/api/projects'),('POST','/api/projects'),('GET','/api/jobs'),('GET','/api/backup'),('GET','/api/readiness'),('POST','/api/evaluations')]:
            self.assertEqual(call(app,method,path,{} if method=='POST' else None).status,401)
        p = self.api('POST','/api/projects',{'title':'Orion','private':False}).json()
        self.assertEqual(self.api('PATCH','/api/projects/'+p['id'],{'version':1,'goal':'Version 2'}).status,200)
        self.assertEqual(self.api('PATCH','/api/projects/'+p['id'],{'version':1,'goal':'Version perdue'}).status,409)
        self.assertEqual(self.api('GET','/api/readiness').json()['durable_database'],db.get_db().kind == 'postgres')

    def test_backup_and_private_provider_changes_require_recent_password(self):
        db.get_db().update('owner_sessions',self.session['id'],{'reauth_at':0})
        self.assertEqual(self.api('GET','/api/backup').status,428)
        self.assertEqual(self.api('POST','/api/settings/trusted-providers',{'providers':[]}).status,428)
