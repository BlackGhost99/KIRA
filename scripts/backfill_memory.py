"""Indexe les anciens souvenirs par petits lots, en ligne.

Render Shell : python -m scripts.backfill_memory --limit 50
Une nouvelle exécution reprend les souvenirs non indexés ; aucune clé dans les arguments.
"""
import argparse
import json

from app import db, memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    db.init_db()
    print(json.dumps({"status": memory.semantic_status(), "result": memory.backfill(args.limit)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
