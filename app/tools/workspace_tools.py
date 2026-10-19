"""Lecture des projets et des documents choisis par le propriétaire."""
from .. import workspace
from .registry import register


@register('project_status', 'Relit le projet sélectionné : objectif, tâches et progression des révisions.',
          {'type':'object','properties':{}}, 'Lecture du projet')
def project_status(args, ctx):
    if not ctx.project_id:
        return 'Aucun projet sélectionné. Le propriétaire peut en choisir un dans Projets.'
    return workspace.prompt(ctx.project_id, '')


@register('search_documents', 'Cherche un passage dans les documents du projet sélectionné et renvoie sa page ou section à citer.',
          {'type':'object','properties':{'query':{'type':'string'}},'required':['query']}, 'Recherche dans les documents')
def search_documents(args, ctx):
    if not ctx.project_id:
        return 'Aucun projet sélectionné.'
    pages = workspace.excerpts(ctx.project_id, args.get('query', ''))
    ctx.tainted = ctx.tainted or bool(pages)
    return '\n\n'.join(f"[Document {p['name']} · {p['location']} {p['number']}] /api/documents/{p['document_id']}/pages/{p['number']}\n{p['content'][:2500]}" for p in pages) or 'Aucun passage pertinent. Ne fabrique pas de citation.'
