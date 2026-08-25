# datasette_plugin.py
"""Datasette plugin for the ledger UI.

Registered via the [project.entry-points.datasette] entry point, so any
Datasette launched in an env with lm-eval-ledger installed picks it up
automatically - no flags needed.
"""
from __future__ import annotations

from datasette import hookimpl


@hookimpl
def permission_allowed(datasette, actor, action):
    """Hide the custom-SQL box (and block ?sql= URLs): the ledger UI is
    tables and curated pages. Re-enable free-form SQL via metadata:

        plugins:
          lm-eval-ledger:
            allow_sql: true
    """
    if action == "execute-sql":
        config = datasette.plugin_config("lm-eval-ledger") or {}
        if not config.get("allow_sql"):
            return False
    return None  # no opinion on other permissions
