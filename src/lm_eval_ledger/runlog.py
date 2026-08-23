# runlog.py
"""Tee stdout/stderr to per-run log files."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


class TeeWriter:
    """Write to multiple destinations (file + original stream)."""

    def __init__(self, original, *files):
        self.original = original
        self.files = files

    def write(self, data):
        self.original.write(data)
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        self.original.flush()
        for f in self.files:
            f.flush()

    def fileno(self):
        """Return file descriptor of the original stream (needed by vLLM)."""
        return self.original.fileno()

    def isatty(self):
        """Return whether original stream is a tty."""
        return self.original.isatty()


class OutputLogger:
    """
    Capture stdout and stderr to log files.

    Creates two log files:
    - stdout_only.log: Only stdout
    - combined.log: Both stdout and stderr
    """

    def __init__(self, logs_dir: Path, timestamp: str):
        self.logs_dir = logs_dir
        self.timestamp = timestamp
        self._stdout_file = None
        self._combined_file = None
        self._original_stdout = None
        self._original_stderr = None

    def start(self):
        self.logs_dir.mkdir(exist_ok=True, parents=True)

        stdout_path = self.logs_dir / f"{self.timestamp}_stdout.log"
        combined_path = self.logs_dir / f"{self.timestamp}_combined.log"

        self._stdout_file = stdout_path.open("w", encoding="utf-8")
        self._combined_file = combined_path.open("w", encoding="utf-8")

        # Write header
        header = f"==== Benchmark Run: {datetime.now().isoformat()} ====\n\n"
        self._stdout_file.write(header)
        self._combined_file.write(header)

        # Save originals
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        # Redirect stdout to: console + stdout_file + combined_file
        sys.stdout = TeeWriter(
            self._original_stdout,
            self._stdout_file,
            self._combined_file,
        )

        # Redirect stderr to: console + combined_file only
        sys.stderr = TeeWriter(
            self._original_stderr,
            self._combined_file,
        )

    def stop(self):
        # Restore original streams
        if self._original_stdout:
            sys.stdout = self._original_stdout
            self._original_stdout = None
        if self._original_stderr:
            sys.stderr = self._original_stderr
            self._original_stderr = None

        # Close files
        if self._stdout_file:
            self._stdout_file.close()
            self._stdout_file = None
        if self._combined_file:
            self._combined_file.close()
            self._combined_file = None

