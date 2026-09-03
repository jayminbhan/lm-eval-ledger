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

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA_VERSION = 1

_SAMPLES_BYTES_EXPR = (
    "COALESCE(length(prompt),0) + COALESCE(length(prompt_full),0) + "
    "COALESCE(length(gold),0) + COALESCE(length(gold_data),0) + "
    "COALESCE(length(responses),0) + COALESCE(length(verifier_verdicts),0)"
)
_SAMPLES_BYTES_UPDATE_ALL = f"""
    UPDATE benchmarks SET samples_bytes = (
        SELECT COALESCE(SUM({_SAMPLES_BYTES_EXPR}), 0)
        FROM samples WHERE samples.benchmark_id = benchmarks.benchmark_id)
"""

DEFAULT_LEDGER_NAME = "ledger.sqlite3"


class LedgerDatabase:
    """Append-only ledger of benchmark runs (see module docstring)."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        # Autocommit (isolation_level=None) for immediate writes; busy_timeout
        # lets concurrent writers retry instead of failing immediately.
        # check_same_thread=False: incremental sample writes arrive from
        # backend worker threads; callers serialize access with a lock.
        self.conn = sqlite3.connect(self.db_path, isolation_level=None,
                                    timeout=60, check_same_thread=False)
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
                config_yaml TEXT,
                source_yaml TEXT
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
                verified_accuracy REAL,
                samples_bytes INTEGER,
                gen_tokens INTEGER,
                gen_seconds REAL
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
                image_ids TEXT,
                extracted TEXT GENERATED ALWAYS AS
                    (json_extract(responses, '$[0].extracted')) VIRTUAL,
                stop_reason TEXT GENERATED ALWAYS AS
                    (json_extract(responses, '$[0].stop_reason')) VIRTUAL
            )
        """)
        # Images referenced by samples, content-addressed by sha256 so a
        # benchmark's images are stored once no matter how many runs use
        # them. Excluded from samples_bytes (shared, not per-benchmark).
        c.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id INTEGER PRIMARY KEY,
                sha256 TEXT UNIQUE NOT NULL,
                mime TEXT,
                data BLOB NOT NULL
            )
        """)
        # Migrations for ledgers created under earlier schema revisions.
        # VIRTUAL generated columns are metadata-only: instant on any size DB.
        for ddl in (
            "ALTER TABLE samples ADD COLUMN gold_data TEXT",
            "ALTER TABLE samples ADD COLUMN image_ids TEXT",
            "ALTER TABLE samples ADD COLUMN extracted TEXT GENERATED ALWAYS AS "
            "(json_extract(responses, '$[0].extracted')) VIRTUAL",
            "ALTER TABLE samples ADD COLUMN stop_reason TEXT GENERATED ALWAYS AS "
            "(json_extract(responses, '$[0].stop_reason')) VIRTUAL",
        ):
            try:
                c.execute(ddl)
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        # model_tag denormalized onto samples (the table view cannot join);
        # backfill runs once, only when the column was just added.
        try:
            c.execute("ALTER TABLE samples ADD COLUMN model_tag TEXT")
            c.execute(
                "UPDATE samples SET model_tag = (SELECT b.model_tag "
                "FROM benchmarks b WHERE b.benchmark_id = samples.benchmark_id)"
            )
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
        # the config file exactly as the user wrote it (resolved config
        # remains in config_yaml)
        try:
            c.execute("ALTER TABLE runs ADD COLUMN source_yaml TEXT")
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
        for ddl in (
            "ALTER TABLE benchmarks ADD COLUMN gen_tokens INTEGER",
            "ALTER TABLE benchmarks ADD COLUMN gen_seconds REAL",
        ):
            try:
                c.execute(ddl)
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        # per-benchmark sample storage footprint, maintained at finalize;
        # one-time backfill for pre-existing ledgers
        try:
            c.execute("ALTER TABLE benchmarks ADD COLUMN samples_bytes INTEGER")
            print("[INFO] Backfilling per-benchmark storage sizes "
                  "(one-time; may take a while on a large ledger)...")
            c.execute(_SAMPLES_BYTES_UPDATE_ALL)
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                raise
        c.execute("CREATE INDEX IF NOT EXISTS idx_benchmarks_run ON benchmarks(run_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_benchmarks_task ON benchmarks(task, model_tag)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_samples_benchmark ON samples(benchmark_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_samples_sample_id ON samples(sample_id)")
        # cross-benchmark joins pair on sample_id; guarantee one row per
        # (benchmark, sample). Existing ledgers with duplicates keep working
        # (plain index) but are warned about.
        try:
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_samples_bench_sample "
                      "ON samples(benchmark_id, sample_id)")
        except sqlite3.IntegrityError:
            print("[WARN] ledger has duplicate (benchmark_id, sample_id) rows; "
                  "pairwise comparisons may double-count those samples")
        # Dropped from the schema; also cleans it out of existing ledgers.
        c.execute("DROP VIEW IF EXISTS samples_flat")
        c.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # ---------- runs ----------

    def create_run(self, run_name: str, config_yaml: str,
                   harness_version: str = "",
                   source_yaml: str | None = None) -> int:
        """Register a new run; returns its run_id.

        config_yaml is the RESOLVED config (reproducible via -c);
        source_yaml is the config file byte-for-byte as the user wrote
        it (None for pure-CLI runs).
        """
        cur = self.conn.execute(
            "INSERT INTO runs (run_name, started_at, harness_version, "
            "config_yaml, source_yaml) VALUES (?, ?, ?, ?, ?)",
            (run_name, datetime.now().isoformat(timespec="seconds"),
             harness_version, config_yaml, source_yaml),
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
                temperature, top_p, max_tokens, error, samples_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
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

    def start_benchmark(self, run_id: int, *, model_tag: str, model: str,
                        task: str, fewshot_k, eval_mode: str, pass_k: int,
                        timestamp: str) -> int:
        """Register an in-progress benchmark so samples can attach to it as
        they complete; accuracy stays NULL until finalize_benchmark."""
        cur = self.conn.execute(
            """INSERT INTO benchmarks (
                run_id, model_tag, model, task, fewshot_k, eval_mode,
                pass_k, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, model_tag, model, task, fewshot_k, eval_mode,
             pass_k, timestamp),
        )
        return cur.lastrowid

    def finalize_benchmark(self, benchmark_id: int, summary: dict) -> None:
        """Fill in the final stats (or error) of an in-progress benchmark."""
        settings = summary.get("settings", {})
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self._finalize_statements(benchmark_id, summary, settings)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def _finalize_statements(self, benchmark_id, summary, settings) -> None:
        self.conn.execute(
            """UPDATE benchmarks SET
                total_examples = ?, correct = ?, accuracy = ?,
                no_answer_count = ?, stop_reason_counts = ?,
                duration_seconds = ?, temperature = ?, top_p = ?,
                max_tokens = ?, error = ?, gen_tokens = ?, gen_seconds = ?
               WHERE benchmark_id = ?""",
            (
                summary.get("total_examples", 0),
                float(summary.get("correct") or 0.0),
                summary.get("accuracy", 0.0),
                summary.get("no_answer_count", 0),
                json.dumps(summary.get("stop_reason_counts") or {}),
                summary.get("duration_seconds"),
                settings.get("temperature"),
                settings.get("top_p"),
                settings.get("max_tokens"),
                summary.get("error"),
                summary.get("gen_tokens"),
                summary.get("gen_seconds"),
                benchmark_id,
            ),
        )
        self.conn.execute(
            f"""UPDATE benchmarks SET samples_bytes = (
                SELECT COALESCE(SUM({_SAMPLES_BYTES_EXPR}), 0)
                FROM samples WHERE benchmark_id = ?)
               WHERE benchmark_id = ?""",
            (benchmark_id, benchmark_id),
        )

    # ---------- samples ----------

    def store_image(self, data: bytes, mime: str = "image/png") -> int:
        """Store one image, deduplicated by content hash; returns image_id."""
        sha = hashlib.sha256(data).hexdigest()
        self.conn.execute(
            "INSERT OR IGNORE INTO images (sha256, mime, data) VALUES (?, ?, ?)",
            (sha, mime, data))
        return self.conn.execute(
            "SELECT image_id FROM images WHERE sha256 = ?", (sha,)
        ).fetchone()["image_id"]

    def add_samples(self, benchmark_id: int, entries: list[dict]) -> None:
        """Add sample rows for one benchmark.

        Each entry: sample_id, prompt, prompt_full, gold (human-readable),
        optional gold_data (machine payload for re-scoring), score, and
        responses = [{"text", "extracted", "stop_reason", "correct"}, ...].
        An optional "_images" key ([(bytes, mime), ...]) is stored in the
        deduplicated images table and recorded as image_ids.
        """
        tag_row = self.conn.execute(
            "SELECT model_tag FROM benchmarks WHERE benchmark_id = ?",
            (benchmark_id,),
        ).fetchone()
        model_tag = tag_row["model_tag"] if tag_row else ""
        for entry in entries:
            imgs = entry.pop("_images", None)
            if imgs:
                entry["image_ids"] = json.dumps(
                    [self.store_image(data, mime) for data, mime in imgs])
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
                entry.get("image_ids"),
            )
            for entry in entries
        ]
        self.conn.executemany(
            "INSERT INTO samples (benchmark_id, model_tag, sample_id, prompt, "
            "prompt_full, gold, gold_data, responses, score, image_ids) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
