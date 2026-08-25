# datasette_plugin.py
"""Datasette plugin for the ledger UI.

Registered via the [project.entry-points.datasette] entry point, so any
Datasette launched in an env with lm-eval-ledger installed picks it up
automatically - no flags needed.

Uses only documented plugin hooks (no template forks, no private APIs),
so it tracks Datasette upgrades:
- permission_allowed: hides the custom-SQL box
- register_routes + extra_css_urls: serves /-/ledger.css with the
  lm-eval-ledger masthead and benchmark-report styling
- menu_links: ledger navigation in the top-right menu
"""
from __future__ import annotations

import re
from urllib.parse import quote

from datasette import hookimpl
from datasette.utils.asgi import Response

_BENCHMARKS_PATH_RE = re.compile(r"^/([^/]+)/benchmarks$")

_LEDGER_CSS = """
/* ---- lm-eval-ledger masthead ---- */
body::before {
  content: "lm-eval-ledger";
  display: block;
  padding: 0.55rem 1rem;
  background: #1f2430;
  color: #e8ecf4;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  border-bottom: 3px solid #4c8bf5;
}

/* ---- benchmark-report table styling ---- */
table.rows-and-columns {
  font-size: 0.85rem;
}
table.rows-and-columns th {
  background: #f0f3f8;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
table.rows-and-columns td {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  vertical-align: top;
}
table.rows-and-columns tr:nth-child(even) td {
  background: #fafbfd;
}

/* links in the datasette header area pick up the accent */
header a, .hd a { color: #4c8bf5; }
"""


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


@hookimpl
def register_routes():
    async def ledger_css(request):
        return Response(
            _LEDGER_CSS, content_type="text/css; charset=utf-8",
            headers={"Cache-Control": "max-age=60"},
        )
    return [(r"^/-/ledger\.css$", ledger_css)]


@hookimpl
def extra_css_urls():
    return ["/-/ledger.css"]


@hookimpl
def asgi_wrapper(datasette):
    """Default view for the benchmarks table: bare requests redirect to a
    per-task leaderboard - faceted by task, first task (alphabetically)
    selected, sorted by accuracy descending. Any explicit query string is
    left untouched, so every other view remains reachable."""
    def wrap(app):
        async def wrapper(scope, receive, send):
            if (scope.get("type") == "http"
                    and scope.get("method") == "GET"
                    and not scope.get("query_string")):
                match = _BENCHMARKS_PATH_RE.match(scope.get("path", ""))
                if match:
                    first_task = None
                    try:
                        db = datasette.get_database(match.group(1))
                        result = await db.execute("SELECT MIN(task) FROM benchmarks")
                        first_task = result.first()[0]
                    except Exception:
                        pass  # not a ledger database; fall through
                    if first_task:
                        location = (
                            f"{scope['path']}?_facet=task"
                            f"&task__exact={quote(first_task)}"
                            f"&_sort_desc=accuracy"
                        )
                        await send({
                            "type": "http.response.start",
                            "status": 302,
                            "headers": [(b"location", location.encode("utf-8"))],
                        })
                        await send({"type": "http.response.body", "body": b""})
                        return
            await app(scope, receive, send)
        return wrapper
    return wrap


@hookimpl
def menu_links(datasette, actor):
    async def inner():
        links = []
        for db_name, db in datasette.databases.items():
            if db_name in ("_internal", "_memory"):
                continue
            try:
                await db.execute("SELECT 1 FROM runs LIMIT 1")
            except Exception:
                continue  # not a ledger database
            base = datasette.urls.database(db_name)
            links += [
                {"href": f"{base}/runs", "label": "Runs"},
                {"href": f"{base}/benchmarks", "label": "Benchmarks"},
                {"href": f"{base}/samples", "label": "Samples"},
            ]
        return links or None
    return inner
