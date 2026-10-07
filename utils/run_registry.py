"""SQLite conversion run registry (#6) with stage tracking for resume (#13).

Tables:
  runs(id, model_id, strategy, bits, status, created_at, finished_at, git_sha, notes)
  metrics(run_id, name, value)
  artifacts(run_id, kind, path)
  stages(run_id, name, status, attempts, updated_at, output)

Default DB: ./mlx_convertion.db (override with MLX_CONVERTION_DB).
"""
from __future__ import annotations

import os
import sqlite3
import subprocess  # nosec B404 - fixed argv, no shell
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional

DEFAULT_DB = Path(os.environ.get("MLX_CONVERTION_DB", "mlx_convertion.db"))

RUN_STATUSES = {"running", "passed", "failed"}
STAGE_STATUSES = {"pending", "running", "done", "failed"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    strategy TEXT,
    bits INTEGER,
    status TEXT NOT NULL DEFAULT 'running',
    created_at TEXT NOT NULL,
    finished_at TEXT,
    git_sha TEXT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (run_id, name)
);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stages (
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    output TEXT,
    PRIMARY KEY (run_id, name)
);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_id, created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def current_git_sha(cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(  # nosec B603 B607 - fixed argv
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class RunRegistry:
    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.db_path = Path(db_path)
        if self.db_path.parent and str(self.db_path.parent) not in ("", "."):
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- runs -----------------------------------------------------------
    def start_run(self, model_id: str, strategy: Optional[str] = None, bits: Optional[int] = None,
                  git_sha: Optional[str] = None, notes: Optional[str] = None) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO runs(model_id, strategy, bits, status, created_at, git_sha, notes) "
                "VALUES (?, ?, ?, 'running', ?, ?, ?)",
                (model_id, strategy, bits, _now(), git_sha, notes),
            )
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str, metrics: Optional[Mapping[str, float]] = None) -> None:
        if status not in RUN_STATUSES - {"running"}:
            raise ValueError(f"invalid final status: {status}")
        with self._conn() as c:
            c.execute("UPDATE runs SET status=?, finished_at=? WHERE id=?", (status, _now(), run_id))
            for name, value in (metrics or {}).items():
                c.execute(
                    "INSERT OR REPLACE INTO metrics(run_id, name, value) VALUES (?, ?, ?)",
                    (run_id, name, None if value is None else float(value)),
                )

    def record_metrics(self, run_id: int, metrics: Mapping[str, float]) -> None:
        with self._conn() as c:
            for name, value in metrics.items():
                c.execute(
                    "INSERT OR REPLACE INTO metrics(run_id, name, value) VALUES (?, ?, ?)",
                    (run_id, name, None if value is None else float(value)),
                )

    def add_artifact(self, run_id: int, kind: str, path: Path | str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO artifacts(run_id, kind, path) VALUES (?, ?, ?)", (run_id, kind, str(path)))

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if row is None:
                return None
            run = dict(row)
            run["metrics"] = {r["name"]: r["value"] for r in c.execute(
                "SELECT name, value FROM metrics WHERE run_id=?", (run_id,))}
            run["artifacts"] = [dict(r) for r in c.execute(
                "SELECT kind, path FROM artifacts WHERE run_id=?", (run_id,))]
            run["stages"] = {r["name"]: dict(r) for r in c.execute(
                "SELECT name, status, attempts, updated_at, output FROM stages WHERE run_id=?", (run_id,))}
            return run

    def list_runs(self, limit: int = 20, model_id: Optional[str] = None,
                  status: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM runs"
        clauses, args = [], []
        if model_id:
            clauses.append("model_id=?")
            args.append(model_id)
        if status:
            clauses.append("status=?")
            args.append(status)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(int(limit))
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(q, args)]
            for r in rows:
                r["metrics"] = {m["name"]: m["value"] for m in c.execute(
                    "SELECT name, value FROM metrics WHERE run_id=?", (r["id"],))}
            return rows

    # ---- quality gate hook ---------------------------------------------
    def record_gate_result(self, gate_result: Any, strategy: Optional[str] = None,
                           bits: Optional[int] = None, run_id: Optional[int] = None) -> int:
        """Persist a verification.quality_gate.QualityGateResult (pass OR fail)."""
        if run_id is None:
            run_id = self.start_run(gate_result.model_name, strategy=strategy, bits=bits,
                                    git_sha=current_git_sha())
        metrics: Dict[str, float] = {}
        for gate, ok in (gate_result.gates_passed or {}).items():
            metrics[f"gate_{gate}"] = 1.0 if ok else 0.0
        for key, value in (gate_result.summary or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = float(value)
        self.finish_run(run_id, "passed" if gate_result.passed else "failed", metrics)
        return run_id

    # ---- stages / resume (#13) -----------------------------------------
    def stage_status(self, run_id: int, name: str) -> Optional[str]:
        with self._conn() as c:
            row = c.execute("SELECT status FROM stages WHERE run_id=? AND name=?", (run_id, name)).fetchone()
            return row["status"] if row else None

    def set_stage(self, run_id: int, name: str, status: str, output: Optional[str] = None) -> None:
        if status not in STAGE_STATUSES:
            raise ValueError(f"invalid stage status: {status}")
        with self._conn() as c:
            c.execute(
                "INSERT INTO stages(run_id, name, status, attempts, updated_at, output) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id, name) DO UPDATE SET status=excluded.status, "
                "attempts=stages.attempts + (CASE WHEN excluded.status='running' THEN 1 ELSE 0 END), "
                "updated_at=excluded.updated_at, output=COALESCE(excluded.output, stages.output)",
                (run_id, name, status, 1 if status == "running" else 0, _now(), output),
            )

    def run_stages(self, run_id: int, stages: Iterable[tuple[str, Callable[[], Optional[str]]]],
                   resume: bool = True) -> Dict[str, str]:
        """Execute named stages in order; completed ('done') stages are skipped on resume.

        A stage callable returns an optional output string (e.g. artifact path).
        A raising stage is marked 'failed' and re-raised; re-running retries it.
        """
        outcome: Dict[str, str] = {}
        for name, fn in stages:
            if resume and self.stage_status(run_id, name) == "done":
                outcome[name] = "skipped"
                continue
            self.set_stage(run_id, name, "running")
            try:
                out = fn()
            except Exception:
                self.set_stage(run_id, name, "failed")
                outcome[name] = "failed"
                raise
            self.set_stage(run_id, name, "done", output=out)
            outcome[name] = "done"
        return outcome
