from __future__ import annotations
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional
from agents_driftguard.core.state import AgentState


class SQLiteStateStore:

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS agent_states (
        step_id       INTEGER PRIMARY KEY,
        timestamp     REAL    NOT NULL,
        action        TEXT    NOT NULL DEFAULT '{}',
        observation   TEXT    NOT NULL DEFAULT '',
        memory        TEXT    NOT NULL DEFAULT '[]',
        scratchpad    TEXT    NOT NULL DEFAULT '',
        goal_embedding TEXT   NOT NULL DEFAULT '[]',
        state_embedding TEXT  NOT NULL DEFAULT '[]',
        inputs        TEXT    NOT NULL DEFAULT '[]',
        outputs       TEXT    NOT NULL DEFAULT '[]',
        metadata      TEXT    NOT NULL DEFAULT '{}',
        diverged      INTEGER NOT NULL DEFAULT 0,
        state_hash    TEXT    NOT NULL DEFAULT ''
    )
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(self._CREATE_TABLE)
        self._conn.commit()

    @staticmethod
    def _state_to_row(state: AgentState) -> dict:
        """Convert an ``AgentState`` to a dict of column values."""
        return {
            "step_id": state.step_id,
            "timestamp": state.timestamp,
            "action": json.dumps(state.action),
            "observation": state.observation,
            "memory": json.dumps(state.memory),
            "scratchpad": state.scratchpad,
            "goal_embedding": json.dumps(state.goal_embedding),
            "state_embedding": json.dumps(state.state_embedding),
            "inputs": json.dumps(state.inputs),
            "outputs": json.dumps(state.outputs),
            "metadata": json.dumps(state.metadata),
            "diverged": int(state.diverged),
            "state_hash": state.state_hash,
        }

    @staticmethod
    def _row_to_state(row: sqlite3.Row) -> AgentState:
        # Handle cases where columns might be missing if migrating (graceful defaults)
        try:
            inputs = json.loads(row["inputs"])
        except (sqlite3.OperationalError, KeyError, IndexError):
            inputs = []
        try:
            outputs = json.loads(row["outputs"])
        except (sqlite3.OperationalError, KeyError, IndexError):
            outputs = []

        return AgentState(
            step_id=row["step_id"],
            timestamp=row["timestamp"],
            action=json.loads(row["action"]),
            observation=row["observation"],
            memory=json.loads(row["memory"]),
            scratchpad=row["scratchpad"],
            goal_embedding=json.loads(row["goal_embedding"]),
            state_embedding=json.loads(row["state_embedding"]),
            inputs=inputs,
            outputs=outputs,
            metadata=json.loads(row["metadata"]),
            diverged=bool(row["diverged"]),
            state_hash=row["state_hash"],
        )

    def save(self, state: AgentState) -> None:
        row = self._state_to_row(state)
        cols = ", ".join(row.keys())
        placeholders = ", ".join(f":{k}" for k in row.keys())
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO agent_states ({cols}) VALUES ({placeholders})",
                row,
            )
            self._conn.commit()

    def get(self, step_id: int) -> Optional[AgentState]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM agent_states WHERE step_id = ?", (step_id,)
            )
            row = cursor.fetchone()
            return self._row_to_state(row) if row else None

    def get_all(self) -> list[AgentState]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM agent_states ORDER BY step_id"
            )
            return [self._row_to_state(row) for row in cursor.fetchall()]

    def get_last_n(self, n: int) -> list[AgentState]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM agent_states ORDER BY step_id DESC LIMIT ?", (n,)
            )
            rows = cursor.fetchall()
            return [self._row_to_state(row) for row in reversed(rows)]

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM agent_states")
            self._conn.commit()

    def truncate_after(self, step_id: int) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM agent_states WHERE step_id > ?", (step_id,)
            )
            count = cursor.fetchone()[0]
            self._conn.execute(
                "DELETE FROM agent_states WHERE step_id > ?", (step_id,)
            )
            self._conn.commit()
            return count

    def close(self) -> None:
        self._conn.close()

    def __len__(self) -> int:
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) FROM agent_states")
            return cursor.fetchone()[0]

    def __repr__(self) -> str:
        return f"SQLiteStateStore(db_path={self.db_path!r}, states={len(self)})"
