"""Empreinte du contenu montré au propriétaire avant son accord."""
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def action_digest(row):
    from .db import jload
    return digest({k: row[k] for k in ('id', 'device_id', 'kind', 'created_at')} | {'args': jload(row['args'], {})})


def proposal_digest(row):
    from .db import jload
    return digest({k: row.get(k) for k in ('title', 'rationale', 'diff', 'tests_ok')} |
                  {k: jload(row.get(k), {}) for k in ('changes', 'artifact', 'evaluation')})
