#!/usr/bin/env python3
"""
VaultStore: store/retrieve-only SQLite vault for entity maps.

- No anonymization logic here; bring your own maps.
- Safe for multi-module, multi-thread usage (WAL mode; per-thread conns; serialized writes).
- Optional at-rest encryption of JSON blobs using Fernet if ENCRYPTION_KEY is provided.

Typical usage
-------------
from vault_store import VaultStore

vault = VaultStore(db_path="./data/vault.db", encryption_key=os.getenv("ENCRYPTION_KEY"))

# Save a map
vault.save_map(
    session_id="sess-1",
    map_id="abc123def456",
    entity_map={"NAME": {"Abhi": "<<NAME_a1b2c3d4e5>>"}},
    reverse_map={"<<NAME_a1b2c3d4e5>>": "Abhi"},
)

# Retrieve later
entity_map, reverse_map = vault.load_map(session_id="sess-1", map_id="abc123def456")

# Inspect
print(vault.list_maps("sess-1"))

# Delete
vault.delete_map("sess-1", "abc123def456")

Concurrency notes
-----------------
- SQLite in WAL mode allows many readers and one writer at a time.
- This class opens one connection per thread (no sharing), and uses a process-wide write lock to serialize writes.
- Keep transactions short; long-running writes will block other writers until commit.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from cryptography.fernet import Fernet  # optional
except Exception:  # cryptography not installed
    Fernet = None  # type: ignore


# -------------------------- Datamodels -------------------------- #

@dataclass(frozen=True)
class VaultRecord:
    session_id: str
    map_id: str
    created_at: str  # ISO8601 UTC


# ----------------------- VaultStore (DB only) ------------------- #

class VaultStore:
    """Minimal, dependency-light store/retrieve vault for entity maps.

    Each thread gets its own SQLite connection (check_same_thread=False),
    and writes are serialized with a global lock to avoid interleaving.
    """

    _SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS entity_maps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        map_id TEXT NOT NULL,
        entity_map_json BLOB NOT NULL,
        reverse_map_json BLOB NOT NULL,
        created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        UNIQUE(session_id, map_id) ON CONFLICT REPLACE
    );
    CREATE INDEX IF NOT EXISTS idx_maps_session ON entity_maps(session_id);
    CREATE INDEX IF NOT EXISTS idx_maps_mapid ON entity_maps(map_id);
    """

    def __init__(
        self,
        db_path: str = "./data/vault.db",
        *,
        encryption_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Connection settings
        self._timeout = float(timeout)
        self._local = threading.local()  # per-thread conn cache
        self._write_lock = threading.RLock()  # serialize writes across threads

        # Optional encryption
        self._fernet = None
        if encryption_key:
            if Fernet is None:
                raise RuntimeError(
                    "cryptography not installed; cannot use encryption. `pip install cryptography`.")
            self._fernet = Fernet(encryption_key)

        # Ensure DB exists + schema applied once using a temp bootstrap connection
        self._bootstrap()

    # -------------------- Connection management -------------------- #

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=self._timeout,
            isolation_level=None,  # autocommit; we'll control transactions with BEGIN/COMMIT
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        # Pragmas to improve concurrency & safety
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(f"PRAGMA busy_timeout={int(self._timeout * 1000)};")
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    def _bootstrap(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self._SCHEMA_SQL)

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None

    # -------------------- Encoding helpers -------------------- #

    def _encode(self, obj: Any) -> bytes:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        if self._fernet:
            return self._fernet.encrypt(raw)
        return raw

    def _decode(self, blob: bytes) -> Any:
        data = blob
        if self._fernet:
            data = self._fernet.decrypt(blob)
        return json.loads(data.decode("utf-8"))

    # -------------------- Public API -------------------- #

    def save_map(
        self,
        *,
        session_id: str,
        map_id: str,
        entity_map: Dict[str, Dict[str, str]],
        reverse_map: Dict[str, str],
        created_at: Optional[str] = None,
    ) -> None:
        """Insert or replace an entity map for (session_id, map_id).

        Uses a global write lock to serialize commits across threads/process threads.
        """
        ts = created_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        conn = self._get_conn()
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE;")  # take write lock early to reduce SQLITE_BUSY
            try:
                conn.execute(
                    """
                    INSERT INTO entity_maps(session_id, map_id, entity_map_json, reverse_map_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, map_id) DO UPDATE SET
                        entity_map_json=excluded.entity_map_json,
                        reverse_map_json=excluded.reverse_map_json,
                        created_at=excluded.created_at
                    """,
                    (session_id, map_id, self._encode(entity_map), self._encode(reverse_map), ts),
                )
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise

    def load_map(self, *, session_id: str, map_id: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
        """Return (entity_map, reverse_map) for given keys or raise KeyError."""
        conn = self._get_conn()
        cur = conn.execute(
            """
            SELECT entity_map_json, reverse_map_json
            FROM entity_maps
            WHERE session_id=? AND map_id=?
            ORDER BY id DESC
            LIMIT 1
            """,
            (session_id, map_id),
        )
        row = cur.fetchone()
        if not row:
            raise KeyError(f"No entity_map for session_id={session_id} map_id={map_id}")
        return self._decode(row[0]), self._decode(row[1])

    def list_maps(self, session_id: str, limit: int = 50) -> List[VaultRecord]:
        conn = self._get_conn()
        cur = conn.execute(
            """
            SELECT session_id, map_id, created_at
            FROM entity_maps
            WHERE session_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, int(limit)),
        )
        return [VaultRecord(r["session_id"], r["map_id"], r["created_at"]) for r in cur.fetchall()]

    def delete_map(self, session_id: str, map_id: str) -> int:
        conn = self._get_conn()
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cur = conn.execute(
                    "DELETE FROM entity_maps WHERE session_id=? AND map_id=?",
                    (session_id, map_id),
                )
                changes = cur.rowcount or 0
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise
        return changes

    def purge_session(self, session_id: str) -> int:
        conn = self._get_conn()
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                cur = conn.execute(
                    "DELETE FROM entity_maps WHERE session_id=?",
                    (session_id,),
                )
                changes = cur.rowcount or 0
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise
        return changes

    def healthcheck(self) -> bool:
        conn = self._get_conn()
        cur = conn.execute("SELECT 1")
        return cur.fetchone() is not None

    # -------------------- Utilities -------------------- #

    @staticmethod
    def generate_fernet_key() -> str:
        """Return a new Fernet key for at-rest encryption (requires cryptography)."""
        if Fernet is None:
            raise RuntimeError("cryptography not installed; cannot generate keys.")
        return Fernet.generate_key().decode()


# -------------------------- Demo CLI -------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VaultStore demo (store/retrieve only)")
    parser.add_argument("--db", default="./demo_vault.db")
    parser.add_argument("--session", default="sess-1")
    parser.add_argument("--map", dest="map_id", default="demo-001")
    parser.add_argument("--encrypt", dest="encrypt", default=None, help="Fernet key (optional)")
    args = parser.parse_args()

    vault = VaultStore(db_path=args.db, encryption_key=args.encrypt)

    # Save sample
    entity_map = {"NAME": {"Alice": "<<NAME_xxx000>>"}}
    reverse_map = {"<<NAME_xxx000>>": "Alice"}
    vault.save_map(session_id=args.session, map_id=args.map_id, entity_map=entity_map, reverse_map=reverse_map)

    # Load
    em, rm = vault.load_map(session_id=args.session, map_id=args.map_id)
    print("Loaded entity_map:", em)
    print("Loaded reverse_map:", rm)

    # List
    print("Recent maps:", vault.list_maps(args.session))

    # Delete example (comment if you want to keep it)
    # print("Deleted:", vault.delete_map(args.session, args.map_id))
