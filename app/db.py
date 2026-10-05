"""Couche base de données : SQLite en local et dans les tests, PostgreSQL (Supabase) en ligne.

Volontairement fine (pas d'ORM) : du SQL portable avec des « ? » comme paramètres.
Les dates sont des chaînes ISO 8601 en UTC, le JSON est stocké en texte.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Iterable

from . import config


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def jdump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def jload(text: Any, default: Any = None) -> Any:
    if text is None or text == "":
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


SCHEMA = [
    """CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS messages (
        id {PK},
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        meta TEXT NOT NULL DEFAULT '{}',
        feedback INTEGER,
        feedback_note TEXT,
        created_at TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS ix_messages_conv ON messages(conversation_id, id)",
    """CREATE TABLE IF NOT EXISTS memories (
        id {PK},
        kind TEXT NOT NULL,
        content TEXT NOT NULL,
        tags TEXT NOT NULL DEFAULT '',
        source TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS ix_memories_kind ON memories(kind)",
    """CREATE TABLE IF NOT EXISTS veille_sources (
        id {PK},
        url TEXT NOT NULL UNIQUE,
        name TEXT NOT NULL DEFAULT '',
        kind TEXT NOT NULL DEFAULT 'rss',
        score REAL NOT NULL DEFAULT 0.5,
        active INTEGER NOT NULL DEFAULT 1,
        last_run TEXT,
        last_error TEXT)""",
    """CREATE TABLE IF NOT EXISTS veille_items (
        id {PK},
        source_id INTEGER REFERENCES veille_sources(id) ON DELETE SET NULL,
        title TEXT,
        url TEXT,
        published TEXT,
        summary TEXT,
        excerpt TEXT,
        topic TEXT,
        level TEXT,
        score REAL,
        status TEXT NOT NULL DEFAULT 'pending',
        summarized_by TEXT,
        content_hash TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL)""",
    "CREATE INDEX IF NOT EXISTS ix_veille_items_status ON veille_items(status)",
    """CREATE TABLE IF NOT EXISTS proposals (
        id {PK},
        title TEXT NOT NULL,
        rationale TEXT NOT NULL DEFAULT '',
        trigger_kind TEXT NOT NULL DEFAULT 'manual',
        changes TEXT NOT NULL DEFAULT '[]',
        diff TEXT NOT NULL DEFAULT '',
        tests_ok INTEGER,
        tests_output TEXT,
        status TEXT NOT NULL DEFAULT 'draft',
        branch TEXT,
        pr_url TEXT,
        error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS llm_usage (
        id {PK},
        ts TEXT NOT NULL,
        day TEXT NOT NULL,
        provider TEXT NOT NULL,
        model TEXT NOT NULL,
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        purpose TEXT NOT NULL DEFAULT 'chat')""",
    "CREATE INDEX IF NOT EXISTS ix_llm_usage_day ON llm_usage(day)",
    """CREATE TABLE IF NOT EXISTS audit (
        id {PK},
        ts TEXT NOT NULL,
        actor TEXT NOT NULL,
        action TEXT NOT NULL,
        details TEXT NOT NULL DEFAULT '{}')""",
    "CREATE INDEX IF NOT EXISTS ix_audit_action ON audit(action)",
    """CREATE TABLE IF NOT EXISTS files (
        id TEXT PRIMARY KEY,
        mime TEXT NOT NULL,
        name TEXT NOT NULL,
        data {BLOB} NOT NULL,
        created_at TEXT NOT NULL)""",
    "CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)",
    # --- le « cœur » : appareils de Brice, codes d'appairage et file d'actions ---
    """CREATE TABLE IF NOT EXISTS devices (
        id {PK},
        name TEXT NOT NULL,
        platform TEXT NOT NULL DEFAULT '',
        token_hash TEXT NOT NULL UNIQUE,
        caps TEXT NOT NULL DEFAULT '{}',
        policy TEXT NOT NULL DEFAULT '{}',
        paused INTEGER NOT NULL DEFAULT 0,
        revoked INTEGER NOT NULL DEFAULT 0,
        info TEXT NOT NULL DEFAULT '{}',
        agent_version TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        last_seen TEXT)""",
    """CREATE TABLE IF NOT EXISTS device_actions (
        id {PK},
        device_id INTEGER NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
        kind TEXT NOT NULL,
        args TEXT NOT NULL DEFAULT '{}',
        category TEXT NOT NULL,
        risk TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL,
        requested_by TEXT NOT NULL DEFAULT 'kira',
        conversation_id TEXT NOT NULL DEFAULT '',
        decided_by TEXT,
        reason TEXT NOT NULL DEFAULT '',
        result TEXT,
        result_files TEXT NOT NULL DEFAULT '[]',
        created_at TEXT NOT NULL,
        decided_at TEXT,
        started_at TEXT,
        finished_at TEXT)""",
    "CREATE INDEX IF NOT EXISTS ix_device_actions_status ON device_actions(device_id, status)",
    """CREATE TABLE IF NOT EXISTS pair_codes (
        id {PK},
        code_hash TEXT NOT NULL UNIQUE,
        expires_at TEXT NOT NULL,
        used INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL)""",
]

POSTGRES_EXTRA = [
    "CREATE INDEX IF NOT EXISTS ix_memories_fts ON memories USING gin (to_tsvector('french', content))",
    "CREATE INDEX IF NOT EXISTS ix_messages_fts ON messages USING gin (to_tsvector('french', content))",
]


class Database:
    def __init__(self, url: str):
        self.url = url
        self.kind = "postgres" if re.match(r"^postgres(ql)?(\+\w+)?://", url) else "sqlite"
        self._local = threading.local()
        if self.kind == "sqlite":
            path = url.split("sqlite:///", 1)[1] if url.startswith("sqlite:///") else url
            self.path = path or "kira.db"
            folder = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(folder, exist_ok=True)
        else:
            self.path = ""

    # -- connexions (une par thread) ---------------------------------------
    def _connect(self):
        if self.kind == "sqlite":
            conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            try:
                conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            return conn
        import psycopg
        from psycopg.rows import dict_row

        url = re.sub(r"^postgres(ql)?(\+\w+)?://", "postgresql://", self.url)
        return psycopg.connect(
            url, autocommit=True, row_factory=dict_row, prepare_threshold=None, connect_timeout=10
        )

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def _drop(self):
        conn = getattr(self._local, "conn", None)
        self._local.conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def close(self):
        self._drop()

    def _is_conn_error(self, exc: Exception) -> bool:
        if self.kind != "postgres":
            return False
        try:
            import psycopg

            return isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError))
        except ImportError:
            return False

    def _sql(self, sql: str) -> str:
        if self.kind == "postgres":
            return sql.replace("%", "%%").replace("?", "%s")
        return sql

    @staticmethod
    def _clean(row: dict) -> dict:
        for k, v in row.items():
            if isinstance(v, memoryview):
                row[k] = bytes(v)
        return row

    # -- API ---------------------------------------------------------------
    def q(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        """Exécute une requête ; renvoie les lignes (SELECT ou INSERT ... RETURNING)."""
        sql = self._sql(sql)
        for attempt in (0, 1):
            try:
                cur = self._conn().execute(sql, tuple(params))
                if cur.description is None:
                    return []
                return [self._clean(dict(r)) for r in cur.fetchall()]
            except Exception as exc:  # noqa: BLE001
                if attempt == 0 and self._is_conn_error(exc):
                    self._drop()
                    continue
                raise
        return []

    def q1(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        rows = self.q(sql, params)
        return rows[0] if rows else None

    def run(self, sql: str, params: Iterable[Any] = ()) -> int:
        """Exécute une écriture ; renvoie le nombre de lignes touchées."""
        sql = self._sql(sql)
        for attempt in (0, 1):
            try:
                return self._conn().execute(sql, tuple(params)).rowcount
            except Exception as exc:  # noqa: BLE001
                if attempt == 0 and self._is_conn_error(exc):
                    self._drop()
                    continue
                raise
        return 0

    def insert(self, table: str, values: dict) -> Any:
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        rows = self.q(f"INSERT INTO {table} ({cols}) VALUES ({marks}) RETURNING id", list(values.values()))
        return rows[0]["id"]

    def update(self, table: str, row_id: Any, values: dict) -> int:
        if not values:
            return 0
        sets = ", ".join(f"{k} = ?" for k in values)
        return self.run(f"UPDATE {table} SET {sets} WHERE id = ?", [*values.values(), row_id])

    def delete(self, table: str, row_id: Any) -> int:
        return self.run(f"DELETE FROM {table} WHERE id = ?", [row_id])

    # -- petites valeurs clé/valeur ---------------------------------------
    def kv_get(self, key: str, default: str = "") -> str:
        row = self.q1("SELECT v FROM kv WHERE k = ?", [key])
        return row["v"] if row else default

    def kv_set(self, key: str, value: str) -> None:
        self.run(
            "INSERT INTO kv (k, v) VALUES (?, ?) ON CONFLICT (k) DO UPDATE SET v = excluded.v",
            [key, value],
        )

    # -- schéma ------------------------------------------------------------
    def init(self) -> None:
        pk = "INTEGER PRIMARY KEY AUTOINCREMENT" if self.kind == "sqlite" else "BIGSERIAL PRIMARY KEY"
        blob = "BLOB" if self.kind == "sqlite" else "BYTEA"
        for stmt in SCHEMA:
            self.run(stmt.replace("{PK}", pk).replace("{BLOB}", blob))
        if self.kind == "postgres":
            for stmt in POSTGRES_EXTRA:
                try:
                    self.run(stmt)
                except Exception:  # noqa: BLE001 — l'index de recherche est un confort
                    pass


_db: Database | None = None
_lock = threading.Lock()


def get_db() -> Database:
    global _db
    if _db is None:
        with _lock:
            if _db is None:
                _db = Database(config.settings.database_url)
    return _db


def configure(url: str) -> Database:
    """Change de base (utilisé par les tests) et crée le schéma."""
    global _db
    with _lock:
        if _db is not None:
            _db.close()
        _db = Database(url)
    _db.init()
    return _db


def init_db() -> Database:
    db = get_db()
    db.init()
    return db
