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

from html import escape as html_escape

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
  if (!document.querySelector("a.ledger-masthead")) {
    document.body.insertAdjacentHTML(
      "afterbegin", '<a class="ledger-masthead" href="/">lm-eval-ledger</a>');
    document.body.classList.add("lel-masthead");
  }
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
/* Fallback (non-clickable) when JS is unavailable; hidden once the
   clickable anchor is in place (body.lel-masthead). */
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
body.lel-masthead::before { display: none; }
a.ledger-masthead, a.ledger-masthead:visited {
  display: block;
  padding: 0.55rem 1rem;
  background: #1f2430;
  color: #e8ecf4;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 1.05rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  border-bottom: 3px solid #4c8bf5;
  text-decoration: none;
}
a.ledger-masthead:hover { color: #ffffff; background: #262c3a; }

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


async def _first_ledger_db(datasette, requested: str | None):
    """The database to compare in: ?db=... or the first ledger found."""
    if requested:
        return requested, datasette.get_database(requested)
    for db_name, db in datasette.databases.items():
        if db_name in ("_internal", "_memory"):
            continue
        try:
            await db.execute("SELECT 1 FROM runs LIMIT 1")
            return db_name, db
        except Exception:
            continue
    return None, None


def _esc(value, limit: int = 120) -> str:
    text = "" if value is None else str(value)
    if len(text) > limit:
        text = text[:limit] + "..."
    return html_escape(text, quote=True)


async def _compare_page(datasette, request):
    """Pairwise sample comparison: pick a task and two (run, model)
    benchmarks, Apply, and see per-sample regressions/improvements
    aligned by sample_id."""
    db_name, db = await _first_ledger_db(datasette, request.args.get("db"))
    if db is None:
        return Response.text("No ledger database attached", status=404)

    benches = (await db.execute(
        "SELECT benchmark_id, run_id, model_tag, task, fewshot_k, "
        "ROUND(COALESCE(verified_accuracy, accuracy), 4) AS acc "
        "FROM benchmarks WHERE (error IS NULL OR error = '') "
        "ORDER BY task, run_id, model_tag"
    )).rows
    tasks = sorted({b["task"] for b in benches})

    sel_task = request.args.get("task", "")
    sel_a = request.args.get("a", "")
    sel_b = request.args.get("b", "")
    only_diff = request.args.get("diff", "")

    task_opts = ['<option value="">(all tasks)</option>'] + [
        f'<option value="{_esc(t)}"{" selected" if t == sel_task else ""}>{_esc(t)}</option>'
        for t in tasks
    ]
    def bench_opts(selected: str) -> str:
        out = ['<option value="">-- select --</option>']
        for b in benches:
            bid = str(b["benchmark_id"])
            label = (f'run {b["run_id"]} · {b["model_tag"]} · '
                     f'{b["task"]}({b["fewshot_k"]}) · acc {b["acc"]}')
            out.append(
                f'<option value="{bid}" data-task="{_esc(b["task"])}"'
                f'{" selected" if bid == selected else ""}>{_esc(label, 200)}</option>')
        return "\n".join(out)

    result_html = ""
    if sel_a and sel_b:
        diff_clause = (
            "AND COALESCE(sa.verified_score, sa.score) "
            "!= COALESCE(sb.verified_score, sb.score)" if only_diff else "")
        rows = (await db.execute(f"""
            SELECT sa.sample_pk AS pk_a, sb.sample_pk AS pk_b,
                   sa.sample_id, sa.gold,
                   sa.extracted AS ans_a, sb.extracted AS ans_b,
                   COALESCE(sa.verified_score, sa.score) AS score_a,
                   COALESCE(sb.verified_score, sb.score) AS score_b
            FROM samples sa JOIN samples sb ON sa.sample_id = sb.sample_id
            WHERE sa.benchmark_id = ? AND sb.benchmark_id = ? {diff_clause}
            ORDER BY CAST(sa.sample_id AS INTEGER), sa.sample_id
        """, [sel_a, sel_b])).rows
        improved = sum(1 for r in rows if r["score_b"] > r["score_a"])
        regressed = sum(1 for r in rows if r["score_b"] < r["score_a"])
        body = []
        for r in rows:
            if r["score_b"] > r["score_a"]:
                cls, badge = "improved", "IMPROVED"
            elif r["score_b"] < r["score_a"]:
                cls, badge = "regressed", "REGRESSED"
            else:
                cls, badge = "same", "="
            body.append(
                f'<tr class="{cls}">'
                f'<td><a href="{datasette.urls.database(db_name)}/samples/{r["pk_a"]}">'
                f'{_esc(r["sample_id"], 40)}</a></td>'
                f'<td>{_esc(r["gold"])}</td>'
                f'<td>{_esc(r["ans_a"])}</td><td>{_esc(r["ans_b"])}</td>'
                f'<td>{r["score_a"]:g}</td><td>{r["score_b"]:g}</td>'
                f'<td class="badge">{badge}</td></tr>')
        result_html = f"""
        <p class="summary"><strong>{len(rows)}</strong> aligned samples ·
        <span class="imp">{improved} improved</span> ·
        <span class="reg">{regressed} regressed</span> ·
        {len(rows) - improved - regressed} unchanged</p>
        <table><thead><tr><th>sample</th><th>gold</th><th>A answered</th>
        <th>B answered</th><th>A</th><th>B</th><th>&Delta;</th></tr></thead>
        <tbody>{"".join(body)}</tbody></table>"""

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Compare - lm-eval-ledger</title>
<link rel="stylesheet" href="/-/ledger.css">
<style>
body {{ margin: 0; font-family: system-ui, sans-serif; }}
main {{ padding: 1rem; }}
form {{ display: flex; gap: 0.6rem; flex-wrap: wrap; align-items: end;
       background: #f0f3f8; padding: 0.8rem 1rem; border-radius: 6px; }}
label {{ display: flex; flex-direction: column; font-size: 0.75rem;
         text-transform: uppercase; letter-spacing: 0.03em; gap: 0.25rem; }}
select {{ font-family: ui-monospace, monospace; max-width: 26rem; }}
table {{ border-collapse: collapse; margin-top: 1rem; font-size: 0.85rem;
         font-family: ui-monospace, monospace; }}
th, td {{ border: 1px solid #d7dde8; padding: 0.3rem 0.55rem; text-align: left;
          vertical-align: top; }}
th {{ background: #f0f3f8; font-size: 0.72rem; text-transform: uppercase; }}
tr.improved td {{ background: #e9f7ee; }}
tr.regressed td {{ background: #fdecec; }}
.badge {{ font-weight: 700; }}
.imp {{ color: #1a7f37; }} .reg {{ color: #c0322f; }}
.summary {{ margin: 1rem 0 0; }}
</style></head><body class="lel-masthead">
<a class="ledger-masthead" href="/">lm-eval-ledger</a>
<main>
<h1>Pairwise comparison</h1>
<form method="get">
  <label>task<select name="task" id="task-sel">{"".join(task_opts)}</select></label>
  <label>benchmark A (baseline)<select name="a">{bench_opts(sel_a)}</select></label>
  <label>benchmark B (candidate)<select name="b">{bench_opts(sel_b)}</select></label>
  <label><span>&nbsp;</span><span><input type="checkbox" name="diff" value="1"
    {"checked" if only_diff else ""}> only changes</span></label>
  <button type="submit">Apply</button>
</form>
{result_html}
<script>
const taskSel = document.getElementById("task-sel");
const filterOpts = () => {{
  const t = taskSel.value;
  document.querySelectorAll('select[name="a"] option, select[name="b"] option')
    .forEach(o => {{
      if (!o.dataset.task) return;
      o.hidden = t !== "" && o.dataset.task !== t;
    }});
}};
taskSel.addEventListener("change", filterOpts);
filterOpts();
</script>
</main></body></html>"""
    return Response.html(page)


async def _consistency_page(datasette, request):
    """Samples that are always wrong / always right across every benchmark
    (run x model) that evaluated them, grouped by (task, sample_id)."""
    db_name, db = await _first_ledger_db(datasette, request.args.get("db"))
    if db is None:
        return Response.text("No ledger database attached", status=404)

    tasks = [r[0] for r in (await db.execute(
        "SELECT DISTINCT task FROM benchmarks ORDER BY task")).rows]
    benches = (await db.execute(
        "SELECT benchmark_id, run_id, model_tag, task, fewshot_k, "
        "ROUND(COALESCE(verified_accuracy, accuracy), 4) AS acc "
        "FROM benchmarks WHERE (error IS NULL OR error = '') "
        "ORDER BY task, run_id, model_tag"
    )).rows
    mode = request.args.get("mode", "wrong")
    sel_task = request.args.get("task", "")
    sel_ids = [v for v in request.args.getlist("b") if v.isdigit()]

    having = ("MAX(COALESCE(s.verified_score, s.score)) <= 0" if mode == "wrong"
              else "MIN(COALESCE(s.verified_score, s.score)) >= 1")
    task_clause = "AND b.task = ?" if sel_task else ""
    bench_clause = ""
    bench_clause2 = ""
    params: list = []
    if sel_ids:
        marks = ",".join("?" * len(sel_ids))
        bench_clause = f"AND b.benchmark_id IN ({marks})"
        bench_clause2 = f"AND b2.benchmark_id IN ({marks})"
        params += sel_ids
    if sel_task:
        params.append(sel_task)
    params += sel_ids  # for the coverage subquery
    # "always" = the verdict holds in EVERY selected benchmark of the
    # sample's task, and the sample was evaluated by all of them.
    rows = (await db.execute(f"""
        SELECT b.task, s.sample_id,
               COUNT(*) AS n_evals,
               COUNT(DISTINCT b.model_tag) AS n_models,
               MIN(s.sample_pk) AS example_pk,
               MIN(s.gold) AS gold
        FROM samples s JOIN benchmarks b USING (benchmark_id)
        WHERE (b.error IS NULL OR b.error = '') {bench_clause} {task_clause}
        GROUP BY b.task, s.sample_id
        HAVING {having}
           AND COUNT(*) = (
               SELECT COUNT(*) FROM benchmarks b2
               WHERE (b2.error IS NULL OR b2.error = '')
                 AND b2.task = b.task {bench_clause2}
           )
        ORDER BY b.task, CAST(s.sample_id AS INTEGER), s.sample_id
    """, params)).rows

    task_opts = ['<option value="">(all tasks)</option>'] + [
        f'<option value="{_esc(t)}"{" selected" if t == sel_task else ""}>{_esc(t)}</option>'
        for t in tasks
    ]
    bench_opts = []
    for b in benches:
        bid = str(b["benchmark_id"])
        label = (f'run {b["run_id"]} · {b["model_tag"]} · '
                 f'{b["task"]}({b["fewshot_k"]}) · acc {b["acc"]}')
        bench_opts.append(
            f'<option value="{bid}" data-task="{_esc(b["task"])}"'
            f'{" selected" if bid in sel_ids else ""}>{_esc(label, 200)}</option>')
    body = []
    for r in rows:
        body.append(
            f'<tr><td>{_esc(r["task"], 40)}</td>'
            f'<td><a href="{datasette.urls.database(db_name)}/samples/{r["example_pk"]}">'
            f'{_esc(r["sample_id"], 40)}</a></td>'
            f'<td>{r["n_evals"]}</td><td>{r["n_models"]}</td>'
            f'<td>{_esc(r["gold"])}</td></tr>')
    label = "always wrong" if mode == "wrong" else "always right"

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Consistency - lm-eval-ledger</title>
<link rel="stylesheet" href="/-/ledger.css">
<style>
body {{ margin: 0; font-family: system-ui, sans-serif; }}
main {{ padding: 1rem; }}
form {{ display: flex; gap: 0.6rem; flex-wrap: wrap; align-items: end;
       background: #f0f3f8; padding: 0.8rem 1rem; border-radius: 6px; }}
label {{ display: flex; flex-direction: column; font-size: 0.75rem;
         text-transform: uppercase; letter-spacing: 0.03em; gap: 0.25rem; }}
select, input[type=number] {{ font-family: ui-monospace, monospace; }}
table {{ border-collapse: collapse; margin-top: 1rem; font-size: 0.85rem;
         font-family: ui-monospace, monospace; }}
th, td {{ border: 1px solid #d7dde8; padding: 0.3rem 0.55rem; text-align: left;
          vertical-align: top; }}
th {{ background: #f0f3f8; font-size: 0.72rem; text-transform: uppercase; }}
.summary {{ margin: 1rem 0 0; }}
</style></head><body class="lel-masthead">
<a class="ledger-masthead" href="/">lm-eval-ledger</a>
<main>
<h1>Sample consistency</h1>
<form method="get">
  <label>show<select name="mode">
    <option value="wrong"{" selected" if mode == "wrong" else ""}>always wrong</option>
    <option value="right"{" selected" if mode == "right" else ""}>always right</option>
  </select></label>
  <label>task<select name="task" id="task-sel">{"".join(task_opts)}</select></label>
  <label>benchmarks (none selected = all)
    <select name="b" multiple size="6">{"".join(bench_opts)}</select></label>
  <button type="submit">Apply</button>
</form>
<p class="summary"><strong>{len(rows)}</strong> samples {label} in every
{"selected benchmark" if sel_ids else "benchmark in the ledger"} covering them
{f"({len(sel_ids)} selected)" if sel_ids else ""}.</p>
<table><thead><tr><th>task</th><th>sample</th><th>evals</th><th>models</th>
<th>gold</th></tr></thead><tbody>{"".join(body)}</tbody></table>
<script>
const taskSel = document.getElementById("task-sel");
const filterOpts = () => {{
  const t = taskSel.value;
  document.querySelectorAll('select[name="b"] option').forEach(o => {{
    o.hidden = t !== "" && o.dataset.task !== t;
  }});
}};
taskSel.addEventListener("change", filterOpts);
filterOpts();
</script>
</main></body></html>"""
    return Response.html(page)


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
    async def compare(datasette, request):
        return await _compare_page(datasette, request)
    async def consistency(datasette, request):
        return await _consistency_page(datasette, request)
    return [
        (r"^/-/ledger\.css$", ledger_css),
        (r"^/-/ledger\.js$", ledger_js),
        (r"^/-/compare$", compare),
        (r"^/-/consistency$", consistency),
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
            links.append({"href": "/-/compare", "label": "Pairwise Compare"})
            links.append({"href": "/-/consistency", "label": "Consistency"})
        return links or None
    return inner
