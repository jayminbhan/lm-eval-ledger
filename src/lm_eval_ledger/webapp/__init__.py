# webapp/__init__.py
"""The ledger viewer: a small, sync, read-only Flask app.

    lm-eval-ledger serve [--db results/ledger.sqlite3] [--port 8090]
                         [--host 127.0.0.1] [--token SECRET]

Design (see README): server-rendered pages over hand-written SQL; the
database is opened read-only per request (fresh snapshot - refreshing
during a run shows newly landed tasks); binds to localhost unless
--host is given; optional shared --token gates all pages for team
sharing behind a reverse proxy. No arbitrary SQL, LIMIT-capped queries,
Jinja autoescaping for untrusted model generations.
"""
from __future__ import annotations

import argparse
import json
import secrets
import sqlite3
import sys
from pathlib import Path

from flask import (Flask, abort, redirect, render_template, request,
                   session, url_for)

PAGE_SIZE = 100

# consistency views cap (samples listed)
CONSISTENCY_LIMIT = 500


def _connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def create_app(db_path: Path, token: str | None = None) -> Flask:
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(32)
    app.config["DB_PATH"] = Path(db_path)

    def q(sql: str, params=()) -> list[sqlite3.Row]:
        con = _connect(app.config["DB_PATH"])
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()

    def q1(sql: str, params=()):
        rows = q(sql, params)
        return rows[0] if rows else None

    # ---------- helpers exposed to templates ----------

    @app.template_filter("trunc")
    def _trunc(value, n=120):
        s = "" if value is None else str(value)
        return s if len(s) <= n else s[:n] + "..."

    @app.template_filter("resp0")
    def _resp0(responses_json, key):
        try:
            r = json.loads(responses_json or "[]")
            return r[0].get(key, "") if r else ""
        except Exception:
            return ""

    # ---------- optional shared-token gate ----------

    if token:
        @app.before_request
        def _gate():
            if request.path.startswith("/static/"):
                return None
            if session.get("ok"):
                return None
            supplied = request.args.get("token") or request.form.get("token")
            if supplied == token:
                session["ok"] = True
                return redirect(request.path)
            return render_template("token.html"), 401

    # ---------- pages ----------

    @app.route("/")
    def runs():
        run_rows = q("SELECT * FROM runs ORDER BY run_id DESC")
        benches = q(
            "SELECT run_id, model_tag, task, fewshot_k, accuracy, "
            "verified_accuracy, total_examples, no_answer_count, error "
            "FROM benchmarks ORDER BY benchmark_id")
        by_run: dict = {}
        for b in benches:
            by_run.setdefault(b["run_id"], []).append(b)
        return render_template("runs.html", runs=run_rows, by_run=by_run)

    @app.route("/benchmarks")
    def benchmarks():
        task = request.args.get("task", "")
        sort = request.args.get("sort", "accuracy")
        if sort not in ("accuracy", "verified_accuracy", "task", "model_tag",
                        "run_id", "duration_seconds"):
            sort = "accuracy"
        dedupe = request.args.get("dedupe") == "1"

        tasks = q("SELECT task, COUNT(*) AS n, "
                  "ROUND(MAX(COALESCE(verified_accuracy, accuracy)), 3) AS best "
                  "FROM benchmarks WHERE (error IS NULL OR error='') "
                  "GROUP BY task ORDER BY task")
        where, params = "", []
        if task:
            where = "AND task = ?"
            params.append(task)
        rows = q(f"""
            SELECT * FROM (
              SELECT b.*, ROW_NUMBER() OVER (
                  PARTITION BY task, model_tag
                  ORDER BY COALESCE(verified_accuracy, accuracy) DESC
              ) AS rn
              FROM benchmarks b WHERE 1=1 {where}
            ) WHERE {("rn = 1" if dedupe else "1=1")}
            ORDER BY {sort} DESC LIMIT 500""", params)
        return render_template("benchmarks.html", rows=rows, tasks=tasks,
                               task=task, sort=sort, dedupe=dedupe)

    def _bench_options():
        return q("SELECT benchmark_id, run_id, model_tag, task, fewshot_k, "
                 "ROUND(COALESCE(verified_accuracy, accuracy), 3) AS acc "
                 "FROM benchmarks WHERE (error IS NULL OR error='') "
                 "ORDER BY task, run_id, model_tag")

    @app.route("/samples")
    def samples():
        mode = request.args.get("mode", "browse")
        ctx = {
            "mode": mode,
            "benches": _bench_options(),
            "tasks": [r["task"] for r in q(
                "SELECT DISTINCT task FROM benchmarks ORDER BY task")],
            "run_list": q("SELECT run_id, run_name FROM runs ORDER BY run_id DESC"),
            "args": request.args,
        }
        if mode == "pairwise":
            return _samples_pairwise(ctx)
        if mode in ("wrong", "right"):
            return _samples_consistency(ctx)
        return _samples_browse(ctx)

    def _samples_browse(ctx):
        where, params = ["1=1"], []
        task = request.args.get("task", "")
        model = request.args.get("model", "")
        bid = request.args.get("benchmark_id", "")
        outcome = request.args.get("outcome", "")
        if task:
            where.append("b.task = ?"); params.append(task)
        if model:
            where.append("s.model_tag = ?"); params.append(model)
        if bid.isdigit():
            where.append("s.benchmark_id = ?"); params.append(int(bid))
        run = request.args.get("run", "")
        if run.isdigit():
            where.append("b.run_id = ?"); params.append(int(run))
        if outcome == "wrong":
            where.append("COALESCE(s.verified_score, s.score) < 1")
        elif outcome == "right":
            where.append("COALESCE(s.verified_score, s.score) >= 1")
        page = max(1, int(request.args.get("page", "1") or 1))
        total = q1(f"SELECT COUNT(*) AS n FROM samples s "
                   f"JOIN benchmarks b USING (benchmark_id) "
                   f"WHERE {' AND '.join(where)}", params)["n"]
        rows = q(f"""
            SELECT s.sample_pk, s.sample_id, s.benchmark_id, s.model_tag,
                   b.task, s.gold, s.extracted, s.stop_reason, s.score,
                   s.verified_score,
                   substr(s.prompt, 1, 500) AS prompt_snip,
                   substr(json_extract(s.responses, '$[0].text'), 1, 1200)
                       AS response_snip
            FROM samples s JOIN benchmarks b USING (benchmark_id)
            WHERE {' AND '.join(where)}
            ORDER BY s.benchmark_id, CAST(s.sample_id AS INTEGER), s.sample_id
            LIMIT ? OFFSET ?""", params + [PAGE_SIZE, (page - 1) * PAGE_SIZE])
        models = [r["model_tag"] for r in q(
            "SELECT DISTINCT model_tag FROM samples ORDER BY 1")]
        ctx.update(rows=rows, total=total, page=page, page_size=PAGE_SIZE,
                   models=models)
        return render_template("samples.html", **ctx)

    def _samples_pairwise(ctx):
        a, b = request.args.get("a", ""), request.args.get("b", "")
        cat = request.args.get("cat", "all")
        if not (a.isdigit() and b.isdigit()):
            ctx.update(pairs=None, cat=cat, a=a, b=b)
            return render_template("samples.html", **ctx)
        label = {str(r["benchmark_id"]):
                 f'run {r["run_id"]} · {r["model_tag"]} · '
                 f'{r["task"]}({r["fewshot_k"]}) · acc {r["acc"]}'
                 for r in ctx["benches"]}
        raw = q("""
            SELECT sa.sample_pk AS pk_a, sb.sample_pk AS pk_b, sa.sample_id,
                   sa.gold, sa.extracted AS ans_a, sb.extracted AS ans_b,
                   COALESCE(sa.verified_score, sa.score) AS score_a,
                   COALESCE(sb.verified_score, sb.score) AS score_b
            FROM samples sa JOIN samples sb ON sa.sample_id = sb.sample_id
            WHERE sa.benchmark_id = ? AND sb.benchmark_id = ?
            ORDER BY CAST(sa.sample_id AS INTEGER), sa.sample_id""",
            [int(a), int(b)])
        pairs = []
        counts = {"improved": 0, "regressed": 0, "answers": 0, "same": 0}
        for r in raw:
            if r["score_b"] > r["score_a"]:
                c = "improved"
            elif r["score_b"] < r["score_a"]:
                c = "regressed"
            elif (r["ans_a"] or "") != (r["ans_b"] or ""):
                c = "answers"
            else:
                c = "same"
            counts[c] += 1
            pairs.append((c, r))
        order = {"improved": 0, "regressed": 1, "answers": 2, "same": 3}
        pairs.sort(key=lambda p: order[p[0]])
        if cat != "all":
            pairs = [p for p in pairs if p[0] == cat]
        swap = dict(request.args, a=b, b=a)
        ctx.update(pairs=pairs, counts=counts, cat=cat, a=a, b=b,
                   label_a=label.get(a, f"benchmark {a}"),
                   label_b=label.get(b, f"benchmark {b}"), swap_args=swap)
        return render_template("samples.html", **ctx)

    def _samples_consistency(ctx):
        mode = ctx["mode"]
        task = request.args.get("task", "")
        sel = [v for v in request.args.getlist("bs") if v.isdigit()]
        having = ("MAX(COALESCE(s.verified_score, s.score)) <= 0"
                  if mode == "wrong"
                  else "MIN(COALESCE(s.verified_score, s.score)) >= 1")
        bench_clause = bench_clause2 = ""
        params: list = []
        if sel:
            marks = ",".join("?" * len(sel))
            bench_clause = f"AND b.benchmark_id IN ({marks})"
            bench_clause2 = f"AND b2.benchmark_id IN ({marks})"
            params += sel
        task_clause = ""
        if task:
            task_clause = "AND b.task = ?"
            params.append(task)
        params += sel
        groups = q(f"""
            SELECT b.task, s.sample_id, COUNT(*) AS n_evals,
                   COUNT(DISTINCT s.model_tag) AS n_models,
                   MIN(s.sample_pk) AS example_pk, MIN(s.gold) AS gold
            FROM samples s JOIN benchmarks b USING (benchmark_id)
            WHERE (b.error IS NULL OR b.error = '') {bench_clause} {task_clause}
            GROUP BY b.task, s.sample_id
            HAVING {having} AND COUNT(*) = (
                SELECT COUNT(*) FROM benchmarks b2
                WHERE (b2.error IS NULL OR b2.error = '')
                  AND b2.task = b.task {bench_clause2})
            ORDER BY b.task, CAST(s.sample_id AS INTEGER), s.sample_id
            LIMIT ?""", params + [CONSISTENCY_LIMIT])
        ctx.update(groups=groups, sel=sel, task=task)
        return render_template("samples.html", **ctx)

    @app.route("/sample/<int:pk>")
    def sample(pk):
        row = q1("""
            SELECT s.*, b.task, b.fewshot_k, b.run_id, b.eval_mode,
                   b.accuracy AS bench_accuracy
            FROM samples s JOIN benchmarks b USING (benchmark_id)
            WHERE s.sample_pk = ?""", [pk])
        if row is None:
            abort(404)
        try:
            responses = json.loads(row["responses"] or "[]")
        except Exception:
            responses = []
        siblings = q("""
            SELECT s.sample_pk, s.model_tag, s.benchmark_id,
                   COALESCE(s.verified_score, s.score) AS sc
            FROM samples s JOIN benchmarks b USING (benchmark_id)
            WHERE b.task = ? AND s.sample_id = ? ORDER BY s.benchmark_id""",
            [row["task"], row["sample_id"]])
        return render_template("sample.html", s=row, responses=responses,
                               siblings=siblings)

    return app


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="lm-eval-ledger serve",
        description="Serve the ledger viewer web app (read-only).")
    p.add_argument("--db", type=Path, default=Path("results/ledger.sqlite3"))
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address (default localhost-only; use "
                        "0.0.0.0 to expose, ideally behind a reverse proxy)")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--token", default=None,
                   help="shared access token gating all pages (team sharing)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)
    if not args.db.exists():
        print(f"[ERROR] Ledger not found: {args.db}", file=sys.stderr)
        sys.exit(1)
    app = create_app(args.db, token=args.token)
    print(f"[SERVE] Ledger viewer on http://{args.host}:{args.port} "
          f"(db: {args.db}, read-only)")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
