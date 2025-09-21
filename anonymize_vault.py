
#!/usr/bin/env python3
"""
anonymize_vault.py
------------------
Standalone DB + anonymization helper for MCP servers.

Purpose
- Provide a tiny, dependency-light module MCP tools can import to:
  1) Anonymize text or a list of message dicts ({"role":?, "content": str})
  2) Store the entity maps in SQLite keyed by session_id and map_id
  3) Later, deanonymize using a provided reverse_map/entity_map or by fetching maps via (session_id, map_id)

Features
- HMAC-SHA256 placeholders: <<ENTITYTYPE_ab12cd34ef>> (10 hex chars, deterministic per (session_id, entity_type, original, secret))
- SQLite schema with indices; WAL mode for concurrency
- Optional at-rest encryption of JSON payloads with Fernet if ENCRYPTION_KEY is set
- Minimal surface area; Presidio is imported lazily only when anonymization is called.

Usage (string):
    from anonymize_vault import AnonymizeVault
    vault = AnonymizeVault(db_path='./data/vault.db', secret='CHANGE_ME')
    anonymized_text, info = vault.anonymize_and_store(
        session_id='sess-1', text='My secret Password is Abc123 and WellsFargo creds here'
    )
    # info: {"map_id", "entity_map", "reverse_map"}

Usage (messages):
    messages = [{"role":"user","content":"Password is Abc123"}, {"role":"assistant","content":"WellsFargo acc"}]
    anonymized_msgs, info = vault.anonymize_and_store(session_id='sess-1', messages=messages)

Deanonymize later:
    original_text = vault.deanonymize(session_id='sess-1', map_id=info["map_id"], text=anonymized_text)
"""
from __future__ import annotations
import base64
import hashlib
import hmac
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None  # type: ignore

Message = Dict[str, Any]
TextOrMessages = Union[str, List[Message]]

@dataclass(frozen=True)
class VaultInfo:
    session_id: str
    map_id: str
    entity_map: Dict[str, Dict[str, str]]
    reverse_map: Dict[str, str]

class AnonymizeVault:
    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS entity_maps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        map_id TEXT NOT NULL,
        entity_map_json BLOB NOT NULL,
        reverse_map_json BLOB NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_maps_session ON entity_maps(session_id);
    CREATE INDEX IF NOT EXISTS idx_maps_mapid ON entity_maps(map_id);
    """

    def __init__(
        self,
        db_path: str = "./data/vault.db",
        secret: Optional[str] = None,
        encryption_key: Optional[str] = None,
    ) -> None:
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.secret = secret or base64.urlsafe_b64encode(os.urandom(32)).decode()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")

        self._fernet = None
        if encryption_key:
            if Fernet is None:
                raise RuntimeError("cryptography not installed; cannot use encryption. `pip install cryptography`.")
            self._fernet = Fernet(encryption_key)

        with self._conn:
            self._conn.executescript(self.SCHEMA_SQL)

    def _placeholder_for(self, entity_type: str, original: str, session_id: str) -> str:
        msg = f"{session_id}|{entity_type}|{original}".encode()
        digest = hmac.new(self.secret.encode(), msg, hashlib.sha256).hexdigest()
        return f"<<{entity_type}_{digest[:10]}>>"

    def _ensure_presidio(self):
        try:
            from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Presidio not available. Install with `pip install presidio-analyzer presidio-anonymizer`"
            ) from e
        return AnalyzerEngine, Pattern, PatternRecognizer

    def _build_analyzer(self):
        AnalyzerEngine, Pattern, PatternRecognizer = self._ensure_presidio()
        analyzer = AnalyzerEngine()
        pass_pattern = Pattern(name="pass_pattern", regex=r"\b\w*(?:p|P)assword\w*\b", score=0.3)
        secr_pattern = Pattern(name="secr_pattern", regex=r"\b\w*(?:s|S)ecret\w*\b", score=0.3)
        cred_pattern = Pattern(name="cred_pattern", regex=r"\b\w*(?:c|C)redential\w*\b", score=0.3)
        org_pattern = Pattern(name="org_pattern", regex=r"\bWells\s*Fargo\b|\bWellsFargo\b", score=0.3)
        passwords_recognizer = PatternRecognizer(supported_entity="NAME1", patterns=[pass_pattern, secr_pattern, cred_pattern, org_pattern])
        analyzer.registry.add_recognizer(passwords_recognizer)
        return analyzer

    @staticmethod
    def _join_messages(messages: List[Message]) -> str:
        return "\u0000".join([str(m.get("content", "")) for m in messages])

    @staticmethod
    def _split_messages(joined: str, messages: List[Message]) -> List[Message]:
        parts = joined.split("\u0000")
        out: List[Message] = []
        for i, m in enumerate(messages):
            nm = dict(m)
            nm["content"] = parts[i] if i < len(parts) else ""
            out.append(nm)
        return out

    def _analyze(self, text: str, language: str) -> List[Any]:
        analyzer = self._build_analyzer()
        results = analyzer.analyze(text=text, language=language)
        return sorted(results, key=lambda r: (r.start, -r.end))

    def _anonymize_text(self, session_id: str, text: str, language: str = "en") -> Tuple[str, Dict[str, Dict[str, str]], Dict[str, str]]:
        results = self._analyze(text, language)
        cursor = 0
        out_parts: List[str] = []
        entity_map: Dict[str, Dict[str, str]] = {}
        for r in results:
            if r.start < cursor:
                continue
            out_parts.append(text[cursor:r.start])
            etype = r.entity_type
            original = text[r.start:r.end]
            emap = entity_map.setdefault(etype, {})
            if original not in emap:
                emap[original] = self._placeholder_for(etype, original, session_id)
            out_parts.append(emap[original])
            cursor = r.end
        out_parts.append(text[cursor:])
        anonymized = "".join(out_parts)
        reverse_map = {v: k for ent in entity_map.values() for k, v in ent.items()}
        return anonymized, entity_map, reverse_map

    @staticmethod
    def _deanonymize_text(text: str, reverse_map: Dict[str, str]) -> str:
        for ph in sorted(reverse_map.keys(), key=len, reverse=True):
            text = text.replace(ph, reverse_map[ph])
        return text

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

    def save_entity_map(self, session_id: str, map_id: str, entity_map: Dict[str, Dict[str, str]], reverse_map: Dict[str, str]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO entity_maps(session_id, map_id, entity_map_json, reverse_map_json, created_at) VALUES(?,?,?,?,?)",
                (session_id, map_id, self._encode(entity_map), self._encode(reverse_map), datetime.utcnow().isoformat()),
            )

    def load_entity_map(self, session_id: str, map_id: str) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str]]:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "SELECT entity_map_json, reverse_map_json FROM entity_maps WHERE session_id=? AND map_id=? ORDER BY id DESC LIMIT 1",
                (session_id, map_id),
            )
            row = cur.fetchone()
        if not row:
            raise KeyError(f"No entity_map for session_id={session_id} map_id={map_id}")
        return self._decode(row[0]), self._decode(row[1])

    def list_maps(self, session_id: str, limit: int = 50) -> List[Tuple[str, str]]:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "SELECT map_id, created_at FROM entity_maps WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
            return [(r[0], r[1]) for r in cur.fetchall()]

    def delete_map(self, session_id: str, map_id: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM entity_maps WHERE session_id=? AND map_id=?",
                (session_id, map_id),
            )
            return cur.rowcount

    def anonymize_and_store(
        self,
        session_id: str,
        text: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        language: str = "en",
        map_id: Optional[str] = None,
    ) -> Tuple[TextOrMessages, Dict[str, Any]]:
        if text is None and messages is None:
            raise ValueError("Provide either `text` or `messages`.")
        is_messages = messages is not None
        joined = self._join_messages(messages) if is_messages else str(text)
        anonymized, entity_map, reverse_map = self._anonymize_text(session_id, joined, language)
        if not map_id:
            mid_src = f"{session_id}|{datetime.utcnow().isoformat()}".encode()
            map_id = hashlib.sha256(mid_src).hexdigest()[:12]
        self.save_entity_map(session_id, map_id, entity_map, reverse_map)
        if is_messages:
            return self._split_messages(anonymized, messages or []), {"map_id": map_id, "entity_map": entity_map, "reverse_map": reverse_map}
        return anonymized, {"map_id": map_id, "entity_map": entity_map, "reverse_map": reverse_map}

    def deanonymize(
        self,
        session_id: str,
        map_id: Optional[str] = None,
        text: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        entity_map: Optional[Dict[str, Dict[str, str]]] = None,
        reverse_map: Optional[Dict[str, str]] = None,
    ) -> TextOrMessages:
        if text is None and messages is None:
            raise ValueError("Provide either `text` or `messages`.")
        if reverse_map is None:
            if entity_map is not None:
                reverse_map = {v: k for ent in entity_map.values() for k, v in ent.items()}
            elif map_id is not None:
                _, reverse_map = self.load_entity_map(session_id, map_id)
            else:
                raise ValueError("Provide reverse_map or entity_map or (session_id + map_id).")
        if messages is not None:
            joined = self._join_messages(messages)
            dean = self._deanonymize_text(joined, reverse_map)
            return self._split_messages(dean, messages)
        return self._deanonymize_text(str(text), reverse_map)

if __name__ == "__main__":
    vault = AnonymizeVault(db_path="./demo_vault.db", secret="supersecret")
    sample = "My Password is XyZ123 and WellsFargo login"
    try:
        anon, info = vault.anonymize_and_store(session_id="sess-1", text=sample)
        print("Anonymized:", anon)
        print("Map ID:", info["map_id"])
        print("Entity Map:", info["entity_map"])
        restored = vault.deanonymize(session_id="sess-1", map_id=info["map_id"], text=anon)
        print("Restored:", restored)
    except RuntimeError as e:
        print("Install Presidio to run demo:", e)
