"""Routes propriétaires pour les projets et la continuité entre appareils."""
import base64
import binascii

from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from . import config, jobs, workspace
from .db import get_db, jload, now_iso


def routes(endpoint, recent, ApiError):
    def projects(ctx):
        if ctx.request.method == 'GET':
            return {'projects': workspace.list_projects()}
        private = ctx.body.get('private', True)
        if not isinstance(private, bool):
            raise ApiError(400, 'Réglage privé invalide.')
        return workspace.create_project(ctx.text('title', 120), ctx.text('goal', 4000), private)

    def detail(ctx):
        if ctx.request.method == 'GET':
            return workspace.detail(ctx.path['id'])
        try:
            return workspace.patch_project(ctx.path['id'], ctx.body, ctx.body.get('version'))
        except workspace.Conflict as exc:
            raise ApiError(409, str(exc))

    def tasks(ctx):
        return workspace.add_task(ctx.path['id'], ctx.text('title', 500))

    def task(ctx):
        try:
            return workspace.patch_task(ctx.path['id'], ctx.text('state', 30), ctx.text('result', 12000), ctx.body.get('version'))
        except workspace.Conflict as exc:
            raise ApiError(409, str(exc))

    def upload(ctx):
        try:
            raw = base64.b64decode(ctx.text('data', 5_400_000), validate=True)
        except (ValueError, binascii.Error):
            raise ApiError(400, 'Fichier encodé incorrectement.')
        return workspace.add_document(ctx.path['id'], ctx.text('name', 200), raw)

    def page(ctx):
        row = get_db().q1('SELECT d.name,p.number,p.location,p.content FROM document_pages p JOIN documents d ON d.id = p.document_id '
                          'WHERE d.id = ? AND p.number = ?', [ctx.path['id'], ctx.id('number')])
        if not row:
            raise ApiError(404, 'Passage introuvable.')
        return row

    def remove_document(ctx):
        from . import audit
        db = get_db()
        with db.transaction():
            row = db.q1('SELECT * FROM documents WHERE id = ?', [ctx.path['id']])
            if not row:
                raise ApiError(404, 'Document introuvable.')
            # Un import en cours conserve son bail ; attendre son arrêt empêche de recréer des pages après effacement.
            job = db.q1("SELECT * FROM background_jobs WHERE name = ? AND state IN ('queued','running')", ['document-' + row['id']])
            if job:
                raise ApiError(409, "Annulez le traitement et attendez son arrêt avant d'effacer ce document.")
            db.delete('documents', row['id'])
            db.delete('files', row['file_id'])
        audit.log('document_deleted', {'id': row['id']}, actor='owner')
        return {'ok': True}

    def cards(ctx):
        return workspace.add_card(ctx.path['id'], ctx.text('question', 2000), ctx.text('answer', 6000), ctx.text('source', 1000))

    def review(ctx):
        try:
            return workspace.review_card(ctx.path['id'], ctx.body.get('remembered'), ctx.body.get('version'))
        except workspace.Conflict as exc:
            raise ApiError(409, str(exc))

    def job_list(ctx):
        return jobs.status()

    def cancel(ctx):
        return {'cancel_requested': jobs.cancel(ctx.path['id'])}

    def resume(ctx):
        row = get_db().q1('SELECT * FROM background_jobs WHERE id = ?', [ctx.path['id']])
        if not row or row['state'] not in ('failed', 'interrupted', 'cancelled'):
            raise ApiError(409, 'Ce traitement ne peut pas être relancé.')
        if row['kind'] not in jobs._handlers:
            raise ApiError(409, 'Ce type de traitement ne peut pas être repris automatiquement.')
        return {'started': jobs.start(row['name'], kind=row['kind'], payload=jload(row['payload'], {}))}

    def trusted(ctx):
        from . import audit
        from .llm import get_router
        recent(ctx)
        names = ctx.body.get('providers')
        if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
            raise ApiError(400, 'Liste de fournisseurs attendue.')
        if set(names) - {p.name for p in get_router().providers}:
            raise ApiError(400, 'Fournisseur inconnu.')
        get_db().kv_set('llm_trusted_providers', ','.join(sorted(set(names))))
        audit.log('trusted_providers_changed', {'providers': names}, actor='owner')
        return {'ok': True}

    def ready(ctx):
        import os
        from . import VERSION, memory
        from .llm import get_router
        return {'version': VERSION, 'revision': os.environ.get('RENDER_GIT_COMMIT', ''),
                'database': get_db().kind, 'durable_database': get_db().kind == 'postgres',
                'providers': get_router().status(), 'semantic': memory.semantic_status(),
                'jobs': jobs.status(), 'server_time': now_iso()}

    def backup_download(ctx):
        from . import audit, backup
        recent(ctx)
        blob = backup.snapshot()
        audit.log('personal_backup_exported', {}, actor='owner')
        return Response(blob, media_type='application/gzip', headers={'Content-Disposition': 'attachment; filename="kira-personal.json.gz"'})

    def evaluations(ctx):
        from . import evaluation
        if ctx.request.method == 'GET':
            return {'runs': evaluation.recent()}
        jobs.register('evaluation', evaluation.run)
        return {'started': jobs.start('evaluation', kind='evaluation')}

    def r(path, fn, methods, limit=None):
        return Route(path, endpoint(fn, body_limit=limit), methods=methods)
    return [r('/api/evaluations', evaluations, ['GET','POST']), r('/api/backup', backup_download, ['GET']), r('/api/projects', projects, ['GET','POST']), r('/api/projects/{id}', detail, ['GET','PATCH']),
            r('/api/projects/{id}/tasks', tasks, ['POST']), r('/api/project-tasks/{id}', task, ['PATCH']),
            r('/api/projects/{id}/documents', upload, ['POST'], 5_500_000),
            r('/api/documents/{id}/pages/{number:int}', page, ['GET']), r('/api/documents/{id}', remove_document, ['DELETE']),
            r('/api/projects/{id}/cards', cards, ['POST']), r('/api/cards/{id}/review', review, ['POST']),
            r('/api/jobs', job_list, ['GET']), r('/api/jobs/{id}/cancel', cancel, ['POST']), r('/api/jobs/{id}/resume', resume, ['POST']),
            r('/api/settings/trusted-providers', trusted, ['POST']), r('/api/readiness', ready, ['GET'])]
