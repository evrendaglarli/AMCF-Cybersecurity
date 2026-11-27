"""v45 R3 Appendix B — canonical DecisionID and local immutable decision ledger.

SQLite WAL stores the decision snapshot *before* action dispatch. Feedback is
accepted only with an explicit confirmation source and a later availability
timestamp. Immutable hash-chained audit entries protect against silent rewriting;
this is a local reference audit, not enterprise WORM certification.
"""
from __future__ import annotations
import hashlib
import json
import math
import sqlite3
from pathlib import Path
from typing import Any

def _jsonable(obj: Any) -> Any:
    # Canonical JSON must be valid RFC JSON, not Python's NaN/Infinity extensions.
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    # Numpy vectors may occur in normalized event features; serialize their
    # values deterministically rather than failing on ndarray.item().
    if hasattr(obj, "tolist") and callable(obj.tolist):
        return _jsonable(obj.tolist())
    if hasattr(obj, "item") and callable(obj.item):
        return _jsonable(obj.item())
    if isinstance(obj, float):
        if math.isnan(obj):
            raise ValueError("NaN cannot be entered in a decision snapshot")
        if math.isinf(obj):
            return "+Infinity" if obj > 0 else "-Infinity"
    if obj is None or isinstance(obj, (str,int,float,bool)):
        return obj
    raise TypeError(f"Unsupported snapshot type: {type(obj).__name__}")

def canonical_bytes(value: Any) -> bytes:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",",":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")

def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


class DecisionLedger:
    """Local reference implementation of the paper's DecisionID/audit contract.

    A file-backed path is needed for persistence; ':memory:' is deliberately a
    test-only ephemeral setting. Database access must be controlled externally.
    """
    def __init__(self, path: str = ":memory:"):
        if path != ":memory:":
            target = Path(path).expanduser()
            target.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS decisions(
          decision_id TEXT PRIMARY KEY, event_ts_ms INTEGER NOT NULL,
          snapshot TEXT NOT NULL, snapshot_hash TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS feedback(
          decision_id TEXT PRIMARY KEY REFERENCES decisions(decision_id),
          outcome TEXT NOT NULL CHECK(outcome IN ('TP','FP')),
          source TEXT NOT NULL, available_at_ms INTEGER NOT NULL,
          feedback_hash TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS controller_state(
          id INTEGER PRIMARY KEY CHECK(id=1), state TEXT NOT NULL,
          state_hash TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS audit(
          serial INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
          object_id TEXT NOT NULL, payload_hash TEXT NOT NULL,
          previous_hash TEXT NOT NULL, entry_hash TEXT NOT NULL);
        """)
        self.connection.commit()

    def _audit(self, kind: str, object_id: str, payload_hash: str) -> None:
        prev = self.connection.execute(
            "SELECT entry_hash FROM audit ORDER BY serial DESC LIMIT 1").fetchone()
        previous_hash = prev[0] if prev else "0"*64
        value = {"kind":kind, "object_id":object_id, "payload_hash":payload_hash,
                 "previous_hash":previous_hash}
        self.connection.execute(
            "INSERT INTO audit(kind,object_id,payload_hash,previous_hash,entry_hash)"
            " VALUES (?,?,?,?,?)",
            (kind,object_id,payload_hash,previous_hash,digest(value)))

    def record(self, snapshot: dict) -> str:
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("event_ts_ms"),int):
            raise ValueError("Snapshot requires integer event_ts_ms")
        body = canonical_bytes(snapshot).decode("utf-8")
        decision_id = hashlib.sha256(body.encode("utf-8")).hexdigest()
        row = self.connection.execute(
            "SELECT snapshot FROM decisions WHERE decision_id=?",(decision_id,)).fetchone()
        if row:
            if row[0] != body:
                raise ValueError("DecisionID collision or snapshot mismatch")
            return decision_id
        with self.connection:
            self.connection.execute(
                "INSERT INTO decisions VALUES (?,?,?,?)",
                (decision_id,snapshot["event_ts_ms"],body,decision_id))
            self._audit("decision", decision_id, decision_id)
        return decision_id

    def read(self, decision_id: str) -> dict:
        row = self.connection.execute(
            "SELECT snapshot FROM decisions WHERE decision_id=?",(decision_id,)).fetchone()
        if row is None:
            raise KeyError("Unknown DecisionID")
        snapshot = json.loads(row[0])
        if digest(snapshot) != decision_id:
            raise ValueError("Corrupted snapshot content")
        return snapshot

    def accepted(self, decision_id: str) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM feedback WHERE decision_id=?", (decision_id,)).fetchone() is not None

    def latest_feedback_ms(self) -> int | None:
        row = self.connection.execute("SELECT MAX(available_at_ms) FROM feedback").fetchone()
        return row[0]

    def write_feedback_and_state(self, *, decision_id: str, outcome: str,
                                 source: str, available_at_ms: int,
                                 state: dict) -> None:
        if outcome not in ("TP","FP"):
            raise ValueError("Only confirmed TP/FP outcomes are supported")
        if source not in ("analyst_confirmed","validated_automation",
                          "red_team_confirmed","incident_closure"):
            raise ValueError("Unverified feedback source")
        if type(available_at_ms) is not int:
            raise TypeError("Confirmation availability time must be integer milliseconds")
        snapshot = self.read(decision_id)
        if available_at_ms < snapshot["event_ts_ms"]:
            raise ValueError("Future-label leakage: confirmed before decision")
        if self.accepted(decision_id):
            raise ValueError("Feedback for DecisionID already accepted")
        latest = self.latest_feedback_ms()
        if latest is not None and available_at_ms < latest:
            raise ValueError("Feedback must arrive in availability-time order")
        feedback_payload = {"decision_id":decision_id,"outcome":outcome,
                            "source":source,"available_at_ms":available_at_ms}
        payload_hash = digest(feedback_payload)
        state_text = canonical_bytes(state).decode("utf-8")
        state_hash = hashlib.sha256(state_text.encode("utf-8")).hexdigest()
        with self.connection:
            self.connection.execute(
                "INSERT INTO feedback VALUES (?,?,?,?,?)",
                (decision_id,outcome,source,available_at_ms,payload_hash))
            self.connection.execute(
                "INSERT INTO controller_state(id,state,state_hash) VALUES(1,?,?) "
                "ON CONFLICT(id) DO UPDATE SET state=excluded.state,"
                "state_hash=excluded.state_hash", (state_text,state_hash))
            self._audit("confirmed_feedback",decision_id,payload_hash)

    def load_state(self) -> dict | None:
        row = self.connection.execute(
            "SELECT state,state_hash FROM controller_state WHERE id=1").fetchone()
        if row is None:
            return None
        if hashlib.sha256(row[0].encode("utf-8")).hexdigest()!=row[1]:
            raise ValueError("Controller state checksum mismatch")
        return json.loads(row[0])

    def verify_chain(self) -> bool:
        prev = "0"*64
        for kind,obj,payload,previous,entry in self.connection.execute(
                "SELECT kind,object_id,payload_hash,previous_hash,entry_hash FROM audit ORDER BY serial"):
            if previous!=prev or entry != digest(
                {"kind":kind,"object_id":obj,"payload_hash":payload,
                 "previous_hash":previous}):
                return False
            prev=entry
        return True

    def close(self):
        self.connection.close()
