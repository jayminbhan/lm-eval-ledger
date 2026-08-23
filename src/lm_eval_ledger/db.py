# db.py
"""Consolidated SQLite results database for benchmark runs."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path


class BenchmarkDatabase:
    """
    Consolidated SQLite database for all benchmark data.

    Tables:
    - summaries: Per-task accuracy, timing, and settings
    - results: Individual example results (one row per sample, with response_1..response_k columns)
    - (gpu_metrics table removed)
    """

    def __init__(self, db_path: Path, pass_k: int = 1):
        self.db_path = db_path
        self.pass_k = pass_k
        # Use autocommit mode (isolation_level=None) for immediate writes
        # busy_timeout lets concurrent writers retry instead of failing immediately
        self.conn = sqlite3.connect(db_path, isolation_level=None, timeout=60)
        self.conn.row_factory = sqlite3.Row
        # WAL mode allows concurrent reads + writes from multiple processes
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        """Create all tables with proper schema."""
        cursor = self.conn.cursor()

        # Summaries table - per-task results with all metadata
        # stop_reason_counts is JSON: {"stop:-": 450, "stop:####": 50, "length:-": 10}
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name TEXT NOT NULL,
                model_tag TEXT NOT NULL,
                total_examples INTEGER,
                correct INTEGER,
                accuracy REAL,
                no_answer_count INTEGER,
                stop_reason_counts TEXT,
                duration_human TEXT,
                pass_k INTEGER,
                temperature REAL,
                top_p REAL,
                max_tokens INTEGER,
                error TEXT,
                model TEXT NOT NULL
            )
        """)

        # Results table - individual example results (one row per sample)
        # For pass@k, responses are stored in response_1, response_2, ... response_k columns
        # extracted_answers and stop_reasons are JSON arrays with all k values
        response_columns = []
        for i in range(1, self.pass_k + 1):
            response_columns.append(f"response_{i} TEXT")
        response_columns_sql = ",\n                ".join(response_columns)

        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                task_name TEXT NOT NULL,
                sample_id TEXT,
                prompt_actual TEXT,
                prompt_full TEXT,
                gold_answer TEXT,
                is_correct INTEGER,
                extracted_answers TEXT,
                stop_reasons TEXT,
                {response_columns_sql}
            )
        """)

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_model ON summaries(model_tag)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_summaries_task ON summaries(task_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_model ON results(model_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_task ON results(task_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_correct ON results(is_correct)")

        self.conn.commit()

    def add_summary(self, summary: dict):
        """Add a task summary record."""
        cursor = self.conn.cursor()
        settings = summary.get("settings", {})
        # Convert stop_reason_counts dict to JSON string
        stop_reason_counts = summary.get("stop_reason_counts", {})
        stop_reason_counts_json = json.dumps(stop_reason_counts) if stop_reason_counts else "{}"
        cursor.execute("""
            INSERT INTO summaries (
                task_name, model_tag, total_examples, correct, accuracy,
                no_answer_count, stop_reason_counts, duration_human,
                pass_k, temperature, top_p, max_tokens, error, model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            summary.get("task", ""),
            summary.get("model_tag", ""),
            summary.get("total_examples", 0),
            summary.get("correct", 0),
            summary.get("accuracy", 0.0),
            summary.get("no_answer_count", 0),
            stop_reason_counts_json,
            summary.get("duration_human", ""),
            summary.get("pass_k", 1),
            settings.get("temperature"),
            settings.get("top_p"),
            settings.get("max_tokens"),
            summary.get("error"),
            summary.get("model", ""),
        ))
        self.conn.commit()

    def add_results(self, task_name: str, model_name: str, entries: list[dict]):
        """Add multiple result entries for a task.

        Each entry should have:
        - sample_id, prompt_actual, prompt_full, gold_answer, is_correct (overall pass@k result)
        - extracted_answers: list of extracted answers for each k (stored as JSON)
        - stop_reasons: list of stop reasons for each k (stored as JSON)
        - responses: list of response strings for each k
        """
        # Build column names and placeholders dynamically based on pass_k
        base_columns = ["model_name", "task_name", "sample_id", "prompt_actual", "prompt_full",
                        "gold_answer", "is_correct", "extracted_answers", "stop_reasons"]
        response_columns = [f"response_{i}" for i in range(1, self.pass_k + 1)]

        all_columns = base_columns + response_columns
        placeholders = ", ".join(["?"] * len(all_columns))
        columns_sql = ", ".join(all_columns)

        rows = []
        for entry in entries:
            responses = entry.get("responses", [])
            # Pad responses so every response_i column gets a value
            response_values = [
                responses[i] if i < len(responses) else ""
                for i in range(self.pass_k)
            ]
            rows.append((
                model_name,
                task_name,
                entry.get("sample_id", ""),
                entry.get("prompt_actual", ""),
                entry.get("prompt_full", ""),
                entry.get("gold_answer", ""),
                1 if entry.get("is_correct") else 0,
                json.dumps(entry.get("extracted_answers", [])),
                json.dumps(entry.get("stop_reasons", [])),
                *response_values,
            ))

        self.conn.executemany(
            f"INSERT INTO results ({columns_sql}) VALUES ({placeholders})", rows
        )
        self.conn.commit()

    def save_run_config(self, config_yaml: str):
        """Record the resolved run configuration (as YAML text) in the database."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS run_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                config_yaml TEXT NOT NULL
            )
        """)
        self.conn.execute(
            "INSERT INTO run_config (created_at, config_yaml) VALUES (?, ?)",
            (datetime.now().isoformat(timespec="seconds"), config_yaml),
        )
        self.conn.commit()

    def close(self, remove_sidecars: bool = False):
        """Close the database connection.

        Args:
            remove_sidecars: Also delete the -wal/-shm files. Only safe when no
                other process still has the database open (e.g., the last close
                of a run) - deleting a live WAL file can corrupt the database.
        """
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            pass  # Another writer is active; SQLite will checkpoint later
        self.conn.close()
        if remove_sidecars:
            for suffix in ("-wal", "-shm"):
                Path(str(self.db_path) + suffix).unlink(missing_ok=True)

