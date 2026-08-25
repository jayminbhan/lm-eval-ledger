# datasette_plugin.py
"""Datasette plugin for the ledger UI.

Registered via the [project.entry-points.datasette] entry point, so any
Datasette launched in an env with lm-eval-ledger installed picks it up
automatically - no flags needed.

Uses only documented plugin hooks (no template forks, no private APIs),
so it tracks Datasette upgrades:
- permission_allowed: hides the custom-SQL box
- register_routes + extra_css_urls/extra_js_urls: serves /-/ledger.css
  (masthead + benchmark-report styling) and /-/ledger.js (feature-style
  display names for the ledger tables)
- menu_links: ledger navigation in the top-right menu

The SQL table names (runs/benchmarks/samples) are untouched - queries,
the CLI, and the JSON API keep working; only the UI labels change.
"""
from __future__ import annotations

from datasette import hookimpl
from datasette.utils.asgi import Response

# UI display names for the ledger tables (SQL names stay unchanged)
_DISPLAY_NAMES = {
    "runs": "Run History",
    "benchmarks": "Benchmark Results",
    "samples": "Sample Inspection",
}

_LEDGER_JS = """
document.addEventListener("DOMContentLoaded", () => {
  const names = {
    "runs": "Run History",
    "benchmarks": "Benchmark Results",
    "samples": "Sample Inspection",
  };
  const relabel = (el) => {
    const t = el.textContent.trim();
    if (names[t]) el.textContent = names[t];
  };
  document.querySelectorAll("h1, h2 a, .db-table a, li a").forEach(relabel);
  for (const [raw, nice] of Object.entries(names)) {
    document.title = document.title.replace(
      new RegExp("(^|[^\\w])" + raw + "([^\\w]|$)"), "$1" + nice + "$2");
  }
});
"""

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
    async def ledger_js(request):
        return Response(
            _LEDGER_JS, content_type="text/javascript; charset=utf-8",
            headers={"Cache-Control": "max-age=60"},
        )
    return [
        (r"^/-/ledger\.css$", ledger_css),
        (r"^/-/ledger\.js$", ledger_js),
    ]


@hookimpl
def extra_css_urls():
    return ["/-/ledger.css"]


@hookimpl
def extra_js_urls():
    return ["/-/ledger.js"]


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
                {"href": f"{base}/{table}", "label": label}
                for table, label in _DISPLAY_NAMES.items()
            ]
        return links or None
    return inner
