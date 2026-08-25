# datasette_plugin.py
"""Datasette plugin: one-click canned queries with recommended parameters.

Registered via the [project.entry-points.datasette] entry point, so any
Datasette launched in an env with lm-eval-ledger installed picks it up
automatically - no flags needed.

Adds ledger-aware links to the action menu (cog icon) of the database page
and of the samples/samples_flat/benchmarks tables: wrong answers for the
newest run, sample flips between the two newest runs, leaderboard, and
format non-compliance - with run ids filled in from the ledger itself.

The canned queries themselves come from datasette-metadata.yaml (-m).
"""
from __future__ import annotations

from datasette import hookimpl

_LEDGER_TABLES = {"samples", "samples_flat", "benchmarks", "runs"}


async def _ledger_links(datasette, database: str) -> list[dict] | None:
    """Build recommended-parameter links for a ledger database, or None."""
    try:
        db = datasette.get_database(database)
        result = await db.execute(
            "SELECT run_id FROM runs ORDER BY run_id DESC LIMIT 2"
        )
    except Exception:
        return None  # not a ledger (no runs table)
    run_ids = [row[0] for row in result.rows]
    if not run_ids:
        return None

    base = datasette.urls.database(database)
    newest = run_ids[0]
    links = [
        {"href": f"{base}/leaderboard",
         "label": "Leaderboard (all runs)"},
        {"href": f"{base}/wrong_answers?run_id={newest}",
         "label": f"Wrong answers (run {newest})"},
        {"href": f"{base}/format_noncompliance",
         "label": "Format non-compliance"},
    ]
    if len(run_ids) > 1:
        prev = run_ids[1]
        links.append(
            {"href": f"{base}/sample_flips?run_a={prev}&run_b={newest}",
             "label": f"Sample flips (run {prev} -> {newest})"}
        )
    return links


@hookimpl
def database_actions(datasette, database):
    async def inner():
        return await _ledger_links(datasette, database)
    return inner


@hookimpl
def table_actions(datasette, database, table):
    async def inner():
        if table not in _LEDGER_TABLES:
            return None
        return await _ledger_links(datasette, database)
    return inner
