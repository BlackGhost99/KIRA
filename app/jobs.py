"""File de tâches durable, baux renouvelés et reprise des traitements idempotents."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Callable

from . import audit
from .db import get_db, jdump, jload, now_iso

_lock = threading.Lock()
_running: dict[str, str] = {}  # compatibilité des outils de test et de diagnostic
_last: dict[str, dict] = {}
_handlers: dict[str, Callable] = {}
LEASE_SECONDS = 120


class Cancelled(Exception):
    pass


class JobContext:
    def __init__(self, jid, owner):
        self.id, self.owner = jid, owner

    def checkpoint(self, progress=0):
        db = get_db()
        row = db.q1("SELECT * FROM background_jobs WHERE id = ?", [self.id])
        if not row or row['lease_owner'] != self.owner or row['cancel_requested']:
            raise Cancelled()
        db.run("UPDATE background_jobs SET progress = ?, lease_until = ?, updated_at = ? WHERE id = ? AND lease_owner = ?",
               [max(0, min(99, int(progress))), time.time() + LEASE_SECONDS, now_iso(), self.id, self.owner])


def register(kind, handler):
    _handlers[kind] = handler


def install_handlers():
    from . import consolidate, evolution, memory, veille
    register('veille', lambda payload, ctx: veille.run())
    register('evolution', lambda payload, ctx: evolution.propose(payload.get('goal', ''), trigger='manual'))
    def daily(payload, ctx):
        from . import devices
        ctx.checkpoint(5)
        result = {'veille': veille.run()}
        ctx.checkpoint(50)
        result['appareils'] = devices.cleanup()
        result['consolidation'] = consolidate.consolidate()
        ctx.checkpoint(85)
        result['memory_index'] = memory.backfill(20)
        return result
    register('daily', daily)
    from . import workspace
    register('document', workspace.index_document)
    from . import evaluation
    register('evaluation', evaluation.run)


def _execute(jid, fn=None):
    db, owner = get_db(), uuid.uuid4().hex
    now = time.time()
    with db.transaction():
        row = db.q1("SELECT * FROM background_jobs WHERE id = ?" + (" FOR UPDATE" if db.kind == 'postgres' else ''), [jid])
        if not row or row['cancel_requested'] or not (row['state'] == 'queued' or row['state'] == 'running' and row['lease_until'] < now):
            return
        db.run("UPDATE background_jobs SET state = 'running', attempts = attempts + 1, lease_owner = ?, lease_until = ?, updated_at = ? WHERE id = ?",
               [owner, now + LEASE_SECONDS, now_iso(), jid])
    stop = threading.Event()
    def heartbeat():
        try:
            while not stop.wait(20):
                db.run("UPDATE background_jobs SET lease_until = ? WHERE id = ? AND lease_owner = ? AND state = 'running'",
                       [time.time() + LEASE_SECONDS, jid, owner])
        finally:
            db.close()
    pulse = threading.Thread(target=heartbeat, daemon=True)
    pulse.start()
    ctx = JobContext(jid, owner)
    try:
        ctx.checkpoint(row['progress'])
        result = fn() if fn else _handlers[row['kind']](jload(row['payload'], {}), ctx)
        ctx.checkpoint(99)
        state, error = 'done', ''
    except Cancelled:
        state, result, error = 'cancelled', {}, ''
    except Exception as exc:
        state, result, error = 'failed', {}, f'{type(exc).__name__}: traitement interrompu'
        from .evolution import EvolutionCapacityError
        from .llm.base import LLMUnavailable
        if isinstance(exc, EvolutionCapacityError):
            error = "Analyse trop volumineuse pour le fournisseur, même en mode compact. Choisissez un objectif plus petit ou un fournisseur disposant de plus de capacité."
        elif isinstance(exc, LLMUnavailable):
            error = "Aucun fournisseur d'IA disponible pour ce traitement. Consultez les réglages des fournisseurs avant de relancer."
        audit.log('job_error', {'job': row['name'], 'type': type(exc).__name__})
    finally:
        stop.set()
        pulse.join(timeout=2)
    latest = db.q1('SELECT progress FROM background_jobs WHERE id = ?', [jid])
    progress = 100 if state == 'done' else (latest['progress'] if latest else 0)
    with db.transaction():
        db.run("UPDATE background_jobs SET state = ?, result = ?, error = ?, progress = ?, lease_until = 0, updated_at = ? "
               "WHERE id = ? AND lease_owner = ?", [state, jdump(result or {}), error, progress, now_iso(), jid, owner])
        audit.log('job_finished', {'id': jid, 'state': state})
    db.close()


def start(name: str, fn: Callable[[], dict] | None = None, wait: bool = False, *, kind=None, payload=None) -> bool:
    if (kind or name) in ('veille', 'evolution', 'daily') and (kind or name) not in _handlers:
        install_handlers()
    db = get_db()
    kind = kind or (name if name in _handlers else 'transient')
    with db.transaction():
        row = db.q1("SELECT * FROM background_jobs WHERE name = ?" + (" FOR UPDATE" if db.kind == 'postgres' else ''), [name])
        if row and row['state'] in ('queued', 'running'):
            return False
        ts = now_iso()
        if row:
            jid = row['id']
            db.update('background_jobs', jid, {'kind': kind, 'payload': jdump(payload or {}), 'state': 'queued',
                      'attempts': 0, 'progress': 0, 'cancel_requested': 0, 'error': '', 'result': '{}', 'updated_at': ts})
        else:
            jid = uuid.uuid4().hex
            # Le conflit UNIQUE protège aussi deux instances qui démarrent le même nom.
            inserted = db.run("INSERT INTO background_jobs (id,name,kind,payload,created_at,updated_at) VALUES (?,?,?,?,?,?) ON CONFLICT (name) DO NOTHING",
                              [jid, name, kind, jdump(payload or {}), ts, ts])
            if not inserted:
                return False
    audit.log('job_queued', {'id': jid, 'kind': kind})
    thread = threading.Thread(target=_execute, args=(jid, fn), name=f'job-{name}', daemon=True)
    thread.start()
    if wait:
        thread.join()
    return True


def recover():
    """Reprise au démarrage ; un traitement externe non idempotent attend une relance explicite."""
    db, resumed = get_db(), []
    for row in db.q("SELECT * FROM background_jobs WHERE state = 'queued' OR (state = 'running' AND lease_until < ?)", [time.time()]):
        if row['kind'] not in _handlers or row['kind'] == 'evolution' and row['state'] == 'running':
            db.update('background_jobs', row['id'], {'state': 'interrupted', 'error': 'Relance manuelle nécessaire.', 'updated_at': now_iso()})
        elif row['cancel_requested']:
            db.update('background_jobs', row['id'], {'state': 'cancelled', 'updated_at': now_iso()})
        else:
            thread = threading.Thread(target=_execute, args=(row['id'],), daemon=True)
            thread.start()
            resumed.append(row['id'])
    return resumed


def cancel(jid):
    changed = get_db().run("UPDATE background_jobs SET cancel_requested = 1, state = CASE WHEN state = 'queued' THEN 'cancelled' ELSE state END, updated_at = ? "
                           "WHERE id = ? AND state IN ('queued','running')", [now_iso(), jid])
    audit.log('job_cancel_requested', {'id': jid}, actor='owner')
    return bool(changed)


def status():
    rows = get_db().q('SELECT * FROM background_jobs ORDER BY updated_at DESC LIMIT 100')
    return {'running': {r['name']: r['created_at'] for r in rows if r['state'] in ('queued', 'running')},
            'last': {r['name']: {'ok': r['state'] == 'done', 'started': r['created_at'], 'finished': r['updated_at'],
                     'result': jload(r['result'], {}), 'error': r['error']} for r in rows if r['state'] not in ('queued', 'running')},
            'items': [{k: r[k] for k in ('id','name','kind','state','attempts','progress','created_at','updated_at','error')} for r in rows]}


def watch(stop):
    try:
        while not stop.wait(20):
            recover()
    finally:
        get_db().close()
