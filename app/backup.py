"""Sauvegarde personnelle vérifiée ; restauration uniquement vers une base vide."""
import base64
import gzip
import hashlib
import json

from .db import get_db, jdump, now_iso

TABLES = ('projects', 'conversations', 'messages', 'memories', 'memory_history', 'files', 'documents', 'document_pages',
          'project_tasks', 'learning_cards', 'veille_sources', 'veille_items', 'proposals', 'llm_usage', 'background_jobs', 'evaluation_runs', 'audit', 'kv')
KV_KEYS = {'kira_identity', 'llm_routing_mode', 'llm_trusted_providers', 'veille_last_run'}
MAX_SIZE = 32_000_000


def _encode(row):
    return {k: {'__kira_blob__': base64.b64encode(bytes(v)).decode()} if isinstance(v, (bytes, memoryview)) else v for k, v in row.items()}


def _checksum(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def snapshot():
    db, tables = get_db(), {}
    with db.transaction():
        if db.kind == 'postgres':
            db.run('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
        size = 0
        for table in TABLES:
            encoded = []
            offset = 0
            key = 'k' if table == 'kv' else 'id'
            while True:
                batch = db.q(f'SELECT * FROM {table} ORDER BY {key} LIMIT ? OFFSET ?', [10, offset])
                if not batch:
                    break
                for row in batch:
                    if table == 'kv' and row['k'] not in KV_KEYS:
                        continue
                    packed = _encode(row)
                    size += len(jdump(packed).encode())
                    if size > MAX_SIZE - 10000:
                        raise ValueError('Sauvegarde personnelle supérieure à 32 Mo ; utilisez une sauvegarde PostgreSQL administrateur.')
                    encoded.append(packed)
                offset += len(batch)
            tables[table] = {'rows': encoded, 'sha256': _checksum(encoded)}
    raw = jdump({'format': 'kira-personal-v1', 'created_at': now_iso(), 'tables': tables}).encode()
    if len(raw) > MAX_SIZE:
        raise ValueError('Sauvegarde personnelle supérieure à 32 Mo ; utilisez une sauvegarde PostgreSQL administrateur.')
    return gzip.compress(raw)


def unpack(blob):
    import io
    if len(blob) > MAX_SIZE:
        raise ValueError('Archive trop volumineuse.')
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as stream:
        raw = stream.read(MAX_SIZE + 1)
    if len(raw) > MAX_SIZE:
        raise ValueError('Archive décompressée trop volumineuse.')
    data = json.loads(raw)
    if data.get('format') != 'kira-personal-v1' or set(data.get('tables', {})) != set(TABLES):
        raise ValueError('Format de sauvegarde incompatible.')
    for table, part in data['tables'].items():
        if not isinstance(part.get('rows'), list) or part.get('sha256') != _checksum(part['rows']):
            raise ValueError('Sauvegarde altérée : ' + table)
    return data


def restore(blob, db):
    """Le chemin de la base cible vient de l'opérateur, jamais d'une requête distante."""
    data = unpack(blob)
    pending_links = []
    with db.transaction():
        if any(db.q1(f'SELECT COUNT(*) AS n FROM {t}')["n"] for t in TABLES):
            raise ValueError('La restauration exige une base vide ; aucune donnée existante ne sera remplacée.')
        for table in TABLES:
            columns = {r['name'] for r in db.q(f'PRAGMA table_info({table})')} if db.kind == 'sqlite' else {
                r['column_name'] for r in db.q('SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() AND table_name = ?', [table])}
            for encoded in data['tables'][table]['rows']:
                if not isinstance(encoded, dict) or set(encoded) - columns:
                    raise ValueError('Colonnes incompatibles : ' + table)
                row = {}
                for k, v in encoded.items():
                    if isinstance(v, dict):
                        if table != 'files' or k != 'data' or set(v) != {'__kira_blob__'}:
                            raise ValueError('Donnée binaire inattendue.')
                        v = base64.b64decode(v['__kira_blob__'], validate=True)
                    if isinstance(v, (dict, list)):
                        raise ValueError('Valeur incompatible.')
                    row[k] = v
                if table == 'kv' and row.get('k') not in KV_KEYS:
                    raise ValueError('Réglage non restaurable.')
                if table == 'memories' and row.get('superseded_by') is not None:
                    pending_links.append((row['id'], row['superseded_by']))
                    row['superseded_by'] = None
                if table == 'background_jobs':
                    row.update(state='interrupted', lease_owner=None, lease_until=0, cancel_requested=0)
                if table == 'proposals':
                    row.update(status='draft', tests_ok=None, approval_hash='', branch=None, pr_url=None)
                names = list(row)
                db.run(f"INSERT INTO {table} ({','.join(names)}) VALUES ({','.join('?' for _ in names)})", list(row.values()))
            if db.kind == 'postgres' and 'id' in columns:
                seq = db.q1('SELECT pg_get_serial_sequence(?, ?) AS s', [table, 'id'])
                if seq and seq['s']:
                    maximum = db.q1(f'SELECT MAX(id) AS n FROM {table}')['n']
                    if maximum is not None:
                        db.q1('SELECT setval(?::regclass, ?, true)', [seq['s'], maximum])
        for mid, target in pending_links:
            db.run('UPDATE memories SET superseded_by = ? WHERE id = ?', [target, mid])
    return {'restored': {t: len(data['tables'][t]['rows']) for t in TABLES}, 'reindex_embeddings': True,
            'reconnect_owner': True, 'pair_devices_again': True}
