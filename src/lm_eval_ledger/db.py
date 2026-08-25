# db.py
"""The ledger: one SQLite database that every benchmark run appends to.

Schema v1 (PRAGMA user_version = 1), three tables:

- runs        one row per harness invocation: name, start time, harness
              version, and the resolved config YAML (full provenance)
- benchmarks  one row per (run, model, task, fewshot) evaluation with all
              scores and metadata; verified_* columns are filled by the
              optional LLM-verifier pass
- samples     one row per evaluated sample. The k pass@k responses live in
              the `responses` column as a JSON array
              [{"text","extracted","stop_reason","correct"}, ...] so the
              schema is identical for every pass_k. `score` is the best
              over k (0/1 for binary tasks, a ratio for partial-credit).

The samples_flat view flattens the first response per sample for quick
inspection in a SQLite browser. Cross-run comparison: `lm-eval-ledger runs`
and `lm-eval-ledger compare`.

Concurrency: WAL mode + autocommit + busy timeout, so multi-GPU workers
(and even concurrent runs) can append to the same ledger safely.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 1

DEFAULT_LEDGER_NAME = "ledger.sqlite3"


class LedgerDatabase:
    """Append-only ledger of benchmark runs (see module docstring)."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        # Autocommit (isolation_level=None) for immediate writes; busy_timeout
        # lets concurrent writers retry instead of failing immediately.
        self.conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=60)
        self.conn.row_factory = sqlite3.Row
        # WAL mode allows concurrent reads + writes from multiple processes
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self) -> None:
        c = self.conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY,
                run_name TEXT NOT NULL,
                started_at TEXT NOT NULL,
                harness_version TEXT,
                config_yaml TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                benchmark_id INTEGER PRIMARY KEY,
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                model_tag TEXT NOT NULL,
                model TEXT NOT NULL,
                task TEXT NOT NULL,
                fewshot_k INTEGER,
                eval_mode TEXT,
                pass_k INTEGER,
                total_examples INTEGER,
                correct REAL,
                accuracy REAL,
                no_answer_count INTEGER,
                stop_reason_counts TEXT,
                duration_seconds REAL,
                timestamp TEXT,
                temperature REAL,
                top_p REAL,
                max_tokens INTEGER,
                error TEXT,
                verifier_model TEXT,
                verifier_mode TEXT,
                verified_correct REAL,
                verified_accuracy REAL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                sample_pk INTEGER PRIMARY KEY,
                benchmark_id INTEGER NOT NULL REFERENCES benchmarks(benchmark_id),
                model_tag TEXT,
                sample_id TEXT,
                prompt TEXT,
                prompt_full TEXT,
                gold TEXT,
                gold_data TEXT,
                responses TEXT,
                score REAL,
                verifier_verdicts TEXT,
                verified_score REAL,
                extracted TEXT GENERATED ALWAYS AS
                    (json_extract(responses, '$[0].extracted')) VIRTUAL,
                stop_reason TEXT GENERATED ALWAYS AS
                    (json_extract(responses, '$[0].stop_reason')) VIRTUAL
            )
        """)
        # Migrations for ledgers created under earlier schema revisions.
        # VIRTUAL generated columns are metadata-only: instant on any size DB.
        for ddl in (
            "ALTER TABLE samples ADD COLUMN gold_data TEXT",
            "ALTER TABLE samples ADD COLUMN extracted TEXT GENERATED ALWAYS AS "
            "(json_extract(responses, '$[0].extracted')) VIRTUAL",
            "ALTER TABLE samples ADD COLUMN stop_reason TEXT GENERATED ALWAYS AS "
            "(json_extract(responses, '$[0].stop_reason')) VIRTUAL",
        ):
            try:
                c.execute(ddl)
            except sqlite3.OperationalError:
                pass  # column already exists
        # model_tag denormalized onto samples (the table view cannot join);
        # backfill runs once, only when the column was just added.
        try:
            c.execute("ALTER TABLE samples ADD COLUMN model_tag TEXT")
            c.execute(
                "UPDATE samples SET model_tag = (SELECT b.model_tag "
                "FROM benchmarks b WHERE b.benchmark_id = samples.benchmark_id)"
            )
        except sqlite3.OperationalError:
            pass  # column already exists (or fresh DB created with it)
        c.execute("CREATE INDEX IF NOT EXISTS idx_benchmarks_run ON benchmarks(run_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_benchmarks_task ON benchmarks(task, model_tag)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_samples_benchmark ON samples(benchmark_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_samples_sample_id ON samples(sample_id)")
        # Dropped from the schema; also cleans it out of existing ledgers.
        c.execute("DROP VIEW IF EXISTS samples_flat")
        c.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # ---------- runs ----------

    def create_run(self, run_name: str, config_yaml: str,
                   harness_version: str = "") -> int:
        """Register a new run; returns its run_id."""
        cur = self.conn.execute(
            "INSERT INTO runs (run_name, started_at, harness_version, config_yaml) "
            "VALUES (?, ?, ?, ?)",
            (run_name, datetime.now().isoformat(timespec="seconds"),
             harness_version, config_yaml),
        )
        return cur.lastrowid

    def get_run_id(self, run_name: str) -> int | None:
        """Look up the newest run with this name (used by multi-GPU workers)."""
        row = self.conn.execute(
            "SELECT run_id FROM runs WHERE run_name = ? ORDER BY run_id DESC LIMIT 1",
            (run_name,),
        ).fetchone()
        return row["run_id"] if row else None

    # ---------- benchmarks ----------

    def add_benchmark(self, run_id: int, summary: dict) -> int:
        """Add one (model, task, fewshot) evaluation; returns benchmark_id."""
        settings = summary.get("settings", {})
        cur = self.conn.execute(
            """INSERT INTO benchmarks (
                run_id, model_tag, model, task, fewshot_k, eval_mode, pass_k,
                total_examples, correct, accuracy, no_answer_count,
                stop_reason_counts, duration_seconds, timestamp,
                temperature, top_p, max_tokens, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                summary.get("model_tag", ""),
                summary.get("model", ""),
                summary.get("task", ""),
                summary.get("fewshot_k"),
                summary.get("eval_mode", ""),
                summary.get("pass_k", 1),
                summary.get("total_examples", 0),
                float(summary.get("correct") or 0.0),
                summary.get("accuracy", 0.0),
                summary.get("no_answer_count", 0),
                json.dumps(summary.get("stop_reason_counts") or {}),
                summary.get("duration_seconds"),
                summary.get("timestamp", ""),
                settings.get("temperature"),
                settings.get("top_p"),
                settings.get("max_tokens"),
                summary.get("error"),
            ),
        )
        return cur.lastrowid

    # ---------- samples ----------

    def add_samples(self, benchmark_id: int, entries: list[dict]) -> None:
        """Add sample rows for one benchmark.

        Each entry: sample_id, prompt, prompt_full, gold (human-readable),
        optional gold_data (machine payload for re-scoring), score, and
        responses = [{"text", "extracted", "stop_reason", "correct"}, ...].
        """
        tag_row = self.conn.execute(
            "SELECT model_tag FROM benchmarks WHERE benchmark_id = ?",
            (benchmark_id,),
        ).fetchone()
        model_tag = tag_row["model_tag"] if tag_row else ""
        rows = [
            (
                benchmark_id,
                model_tag,
                entry.get("sample_id", ""),
                entry.get("prompt", ""),
                entry.get("prompt_full", ""),
                entry.get("gold", ""),
                entry.get("gold_data"),
                json.dumps(entry.get("responses", [])),
                float(entry.get("score") or 0.0),
            )
            for entry in entries
        ]
        self.conn.executemany(
            "INSERT INTO samples (benchmark_id, model_tag, sample_id, prompt, "
            "prompt_full, gold, gold_data, responses, score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    # ---------- lifecycle ----------

    def close(self, remove_sidecars: bool = False) -> None:
        """Close the connection.

        remove_sidecars also deletes the -wal/-shm files; only safe when no
        other process still has the ledger open.
        """
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            pass  # another writer is active; SQLite will checkpoint later
        self.conn.close()
        if remove_sidecars:
            for suffix in ("-wal", "-shm"):
                Path(str(self.db_path) + suffix).unlink(missing_ok=True)
