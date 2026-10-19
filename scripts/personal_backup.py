"""Sauvegarde : python -m scripts.personal_backup export /chemin/backup.json.gz
Restauration : ... restore archive.json.gz --into-sqlite /nouveau/fichier.db
Ou --database-url-env KIRA_RESTORE_DATABASE_URL pour une base PostgreSQL VIDE.
"""
import argparse
import os
from pathlib import Path

from app import backup, db


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('export', 'restore'))
    p.add_argument('file', type=Path)
    target = p.add_mutually_exclusive_group()
    target.add_argument('--into-sqlite', type=Path)
    target.add_argument('--database-url-env')
    a = p.parse_args()
    if a.action == 'export':
        db.init_db()
        blob = backup.snapshot()
        with a.file.open('xb') as out:
            os.chmod(a.file, 0o600)
            out.write(blob)
        print('Sauvegarde créée et vérifiée :', a.file)
        backup.unpack(blob)
    else:
        if a.into_sqlite:
            if a.into_sqlite.exists():
                p.error('Le fichier cible doit être nouveau.')
            url = 'sqlite:///' + str(a.into_sqlite.resolve())
        elif a.database_url_env:
            url = os.environ.get(a.database_url_env, '')
            if not url:
                p.error('Variable de destination vide.')
        else:
            p.error('Indiquez une nouvelle base cible.')
        target_db = db.Database(url)
        target_db.init()
        try:
            result = backup.restore(a.file.read_bytes(), target_db)
            print('Restauration réussie :', result)
        finally:
            target_db.close()


if __name__ == '__main__':
    main()
