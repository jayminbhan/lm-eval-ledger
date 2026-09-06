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

The one exception to read-only: Run History offers per-run / per-task
deletion and database compaction. These are POST-only, confirm-dialog
gated, and behind the same --token gate; a read-write connection is
opened only for those requests.
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


def _connect_rw(db_path: Path) -> sqlite3.Connection:
    """Read-write connection, used only by the delete/compact routes."""
    con = sqlite3.connect(db_path, isolation_level=None, timeout=60)
    con.row_factory = sqlite3.Row
    return con


def _fmt_bytes(n) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def create_app(db_path: Path, token: str | None = None,
               read_only: bool = False) -> Flask:
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(32)
    # the session cookie only carries the --token gate; Strict SameSite
    # means a cross-site form cannot ride it into the POST delete routes
    app.config["SESSION_COOKIE_SAMESITE"] = "Strict"
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["DB_PATH"] = Path(db_path)
    # read_only hard-disables the write routes (delete/compact): they
    # return 403 and their buttons are not rendered. For public
    # deployments (e.g. HF Spaces) where anyone can click anything.
    app.config["READ_ONLY"] = bool(read_only)

    @app.context_processor
    def _flags():
        return {"read_only": app.config["READ_ONLY"]}

    def q(sql: str, params=()) -> list[sqlite3.Row]:
        con = _connect(app.config["DB_PATH"])
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()

    def q1(sql: str, params=()):
        rows = q(sql, params)
        return rows[0] if rows else None

    def _columns(table: str) -> set[str]:
        con = _connect(app.config["DB_PATH"])
        try:
            return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        finally:
            con.close()

    def has_images() -> bool:
        """Old ledgers (not yet opened by the new harness) lack the
        image columns; render without them instead of erroring."""
        return "image_ids" in _columns("samples")

    def _col(table: str, name: str) -> str:
        """SELECT fragment that degrades to NULL on pre-migration
        ledgers (the viewer is read-only and must not require them)."""
        return name if name in _columns(table) else f"NULL AS {name}"

    # ---------- helpers exposed to templates ----------

    @app.template_filter("trunc")
    def _trunc(value, n=120):
        s = "" if value is None else str(value)
        return s if len(s) <= n else s[:n] + "..."

    app.template_filter("fmtbytes")(_fmt_bytes)

    @app.template_filter("fmtdate")
    def _fmtdate(iso):
        """'2026-08-27T21:12:56' -> 'Aug 27, 2026 21:12'."""
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(str(iso))
            # %-d is glibc-only (Windows strftime rejects it); build the
            # day portably instead
            return f"{dt.strftime('%b')} {dt.day}, {dt.year} {dt.strftime('%H:%M')}"
        except (ValueError, TypeError):
            return iso or ""

    @app.template_filter("fmtdur")
    def _fmtdur(seconds):
        if seconds is None:
            return "…"
        s = int(seconds)
        if s >= 3600:
            return f"{s // 3600}h {s % 3600 // 60:02d}m"
        if s >= 60:
            return f"{s // 60}m"
        return "<1m"

    @app.template_filter("imgids")
    def _imgids(image_ids_json):
        try:
            return json.loads(image_ids_json or "[]")
        except Exception:
            return []

    @app.template_filter("split_think")
    def _split_think(text):
        """(thinking, answer) split at the last </think>; thinking is ""
        for non-thinking responses. The opening <think> usually lives in
        the prompt (the template emits it), so only the closing tag is
        matched."""
        t = text or ""
        head, sep, tail = t.rpartition("</think>")
        return (head, tail.lstrip("\n")) if sep else ("", t)

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
            if supplied is not None and secrets.compare_digest(supplied, token):
                session["ok"] = True
                kept = {k: v for k, v in request.args.items() if k != "token"}
                return redirect(url_for(request.endpoint, **request.view_args,
                                        **kept) if request.endpoint else request.path)
            return render_template("token.html"), 401

    # ---------- pages ----------

    @app.route("/")
    def runs():
        run_rows = q(f"SELECT run_id, run_name, started_at, harness_version, "
                     f"config_yaml, {_col('runs', 'source_yaml')} "
                     f"FROM runs ORDER BY run_id DESC")
        benches = q(
            f"SELECT benchmark_id, run_id, model_tag, task, fewshot_k, "
            f"accuracy, verified_accuracy, total_examples, no_answer_count, "
            f"duration_seconds, "
            f"{_col('benchmarks', 'gen_tokens')}, "
            f"{_col('benchmarks', 'gen_seconds')}, "
            f"{_col('benchmarks', 'samples_bytes')}, "
            f"{_col('benchmarks', 'started_at')}, error "
            f"FROM benchmarks ORDER BY benchmark_id")
        # samples_bytes is maintained incrementally by the writer (every
        # flush) and recomputed at finalize - never scanned here: LENGTH()
        # over gigabytes of in-progress responses made this page take
        # seconds. NULL only on rows written before that existed.
        by_run: dict = {}
        run_bytes: dict = {}
        bench_bytes: dict = {}
        for b in benches:
            nbytes = b["samples_bytes"]
            bench_bytes[b["benchmark_id"]] = nbytes
            by_run.setdefault(b["run_id"], []).append(b)
            run_bytes[b["run_id"]] = run_bytes.get(b["run_id"], 0) + (nbytes or 0)
        db_file_bytes = app.config["DB_PATH"].stat().st_size
        # elapsed time for in-progress benchmarks (since their recorded start)
        from datetime import datetime
        elapsed = {}
        for b in benches:
            if b["accuracy"] is None and not b["error"] and b["started_at"]:
                try:
                    elapsed[b["benchmark_id"]] = (
                        datetime.now() - datetime.fromisoformat(b["started_at"])).total_seconds()
                except ValueError:
                    pass
        # backend per run, from the resolved config (per-model overrides
        # may mix backends -> list every distinct one)
        run_backend = {r["run_id"]: _backends_of(r["config_yaml"]) for r in run_rows}
        return render_template("runs.html", runs=run_rows, by_run=by_run, elapsed=elapsed,
                               run_bytes=run_bytes, bench_bytes=bench_bytes,
                               db_file_bytes=db_file_bytes, run_backend=run_backend)

    @app.route("/run/<int:run_id>/config")
    def run_config(run_id):
        row = q1(f"SELECT run_name, config_yaml, "
                 f"{_col('runs', 'source_yaml')} FROM runs "
                 f"WHERE run_id = ?", [run_id])
        if row is None:
            abort(404)
        return render_template("runconfig.html", run_id=run_id,
                               run_name=row["run_name"],
                               source_yaml=row["source_yaml"],
                               config_yaml=row["config_yaml"])

    @app.route("/run/<int:run_id>/config.yaml")
    def run_config_download(run_id):
        kind = request.args.get("kind", "source")
        if kind not in ("source", "resolved"):
            abort(400)
        row = q1(f"SELECT run_name, config_yaml, "
                 f"{_col('runs', 'source_yaml')} FROM runs "
                 f"WHERE run_id = ?", [run_id])
        if row is None:
            abort(404)
        text = row["source_yaml"] if kind == "source" else row["config_yaml"]
        if not text:  # pure-CLI runs have no source file
            text, kind = row["config_yaml"], "resolved"
        if not text:
            abort(404)
        from flask import Response
        return Response(text, mimetype="text/yaml; charset=utf-8", headers={
            "Content-Disposition": f"attachment; filename="
            f"run{run_id}-{kind}.yaml"})

    # ---------- the write routes: delete + compact (POST-only) ----------

    def _rw(statements: list[tuple[str, tuple]]) -> None:
        if app.config["READ_ONLY"]:
            abort(403)
        con = _connect_rw(app.config["DB_PATH"])
        try:
            for sql, params in statements:
                con.execute(sql, params)
        finally:
            con.close()

    @app.route("/run/<int:run_id>/delete", methods=["POST"])
    def delete_run(run_id):
        _rw([
            ("DELETE FROM samples WHERE benchmark_id IN "
             "(SELECT benchmark_id FROM benchmarks WHERE run_id = ?)", (run_id,)),
            ("DELETE FROM benchmarks WHERE run_id = ?", (run_id,)),
            ("DELETE FROM runs WHERE run_id = ?", (run_id,)),
        ])
        return redirect(url_for("runs"))

    @app.route("/benchmark/<int:bid>/delete", methods=["POST"])
    def delete_benchmark(bid):
        _rw([
            ("DELETE FROM samples WHERE benchmark_id = ?", (bid,)),
            ("DELETE FROM benchmarks WHERE benchmark_id = ?", (bid,)),
        ])
        return redirect(url_for("runs"))

    @app.route("/compact", methods=["POST"])
    def compact():
        # Deleted rows free pages inside the file; VACUUM returns them to
        # the filesystem. Heavy on a big ledger - user-invoked only.
        # Images are shared across runs, so they are garbage-collected
        # here (when no remaining sample references them), not on delete.
        stmts = [("VACUUM", ())]
        if has_images():
            stmts.insert(0, (
                "DELETE FROM images WHERE image_id NOT IN "
                "(SELECT value FROM samples, json_each(samples.image_ids) "
                " WHERE samples.image_ids IS NOT NULL)", ()))
        _rw(stmts)
        return redirect(url_for("runs"))

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
        # a leaderboard mixes scores of ONE task; there is no all-tasks
        # view - default to the first task when none is selected
        if not task and tasks:
            task = tasks[0]["task"]
        where, params = "", []
        if task:
            where = "AND task = ?"
            params.append(task)
        rows = [dict(r) for r in q(f"""
            SELECT * FROM (
              SELECT b.*, ROW_NUMBER() OVER (
                  PARTITION BY task, model_tag
                  ORDER BY COALESCE(verified_accuracy, accuracy) DESC
              ) AS rn
              FROM benchmarks b
              WHERE (error IS NULL OR error = '')
                AND accuracy IS NOT NULL {where}
            ) WHERE {("rn = 1" if dedupe else "1=1")}
            ORDER BY {sort} DESC LIMIT 500""", params)]
        # medals for the top 3 by effective accuracy within this task's
        # listing - tied to benchmark_id so they survive re-sorting
        ranked = sorted((r for r in rows), key=lambda r: (
            (r["verified_accuracy"] if r["verified_accuracy"] is not None
             else r["accuracy"]) or 0), reverse=True)
        medals = {r["benchmark_id"]: m
                  for r, m in zip(ranked[:3], ("🥇", "🥈", "🥉"))}
        return render_template("benchmarks.html", rows=rows, tasks=tasks,
                               task=task, sort=sort, dedupe=dedupe,
                               medals=medals)

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
            # a benchmark without an accuracy (in progress / never
            # finished) has nothing to pairwise-compare; newest runs first
            ctx["pair_benches"] = sorted(
                (b for b in ctx["benches"] if b["acc"] is not None),
                key=lambda b: (-b["run_id"], b["benchmark_id"]))
            return _samples_pairwise(ctx)
        if mode in ("wrong", "right"):
            return _samples_consistency(ctx)
        return _samples_browse(ctx)

    def _samples_browse(ctx):
        # One "show" scope instead of four interlocking facet boxes:
        #   b:<id>  one benchmark        t:<task>  a task across runs
        #   r:<id>  everything in a run  m:<tag>   a model across tasks
        # Legacy params (task/model/run/benchmark_id) still filter, so
        # old URLs keep working; active ones render as removable chips.
        scope = request.args.get("scope", "")
        task = request.args.get("task", "")
        model = request.args.get("model", "")
        bid = request.args.get("benchmark_id", "")
        run = request.args.get("run", "")
        outcome = request.args.get("outcome", "")
        invalid_scope = False
        if scope:
            kind, _, val = scope.partition(":")
            ok = ((kind in ("b", "r") and val.isdigit())
                  or (kind in ("t", "m") and bool(val)))
            if ok:
                task = val if kind == "t" else ""
                model = val if kind == "m" else ""
                run = val if kind == "r" else ""
                bid = val if kind == "b" else ""
            else:
                invalid_scope = True

        where, params = ["1=1"], []
        if invalid_scope:
            where.append("0=1")  # a malformed scope must not silently widen to everything
        if task:
            where.append("b.task = ?"); params.append(task)
        if model:
            where.append("s.model_tag = ?"); params.append(model)
        if bid.isdigit():
            where.append("s.benchmark_id = ?"); params.append(int(bid))
        if run.isdigit():
            where.append("b.run_id = ?"); params.append(int(run))
        if outcome == "wrong":
            where.append("COALESCE(s.verified_score, s.score) < 1")
        elif outcome == "right":
            where.append("COALESCE(s.verified_score, s.score) >= 1")

        # scope select contents (benchmarks table is small)
        run_names = {r["run_id"]: r["run_name"] for r in ctx["run_list"]}
        by_run: dict[int, dict] = {}
        for b in ctx["benches"]:
            g = by_run.setdefault(b["run_id"], {
                "run_id": b["run_id"],
                "run_name": run_names.get(b["run_id"], ""), "benches": []})
            g["benches"].append(b)
        groups = sorted(by_run.values(), key=lambda g: -g["run_id"])
        for g in groups:
            g["benches"].sort(key=lambda b: b["benchmark_id"])
        scope_tasks = [r["task"] for r in q(
            "SELECT DISTINCT task FROM benchmarks ORDER BY task")]
        scope_models = [r["model_tag"] for r in q(
            "SELECT DISTINCT model_tag FROM benchmarks ORDER BY model_tag")]

        # the current selection as a scope value (custom combos -> "")
        active = [p for p in (("t", task), ("m", model), ("r", run), ("b", bid))
                  if p[1]]
        scope_value = f"{active[0][0]}:{active[0][1]}" if len(active) == 1 else ""

        # plain-language chips describing every active filter; each chip's
        # "remove" link is the same URL minus that parameter
        def url_without(*keys):
            kept = {k: v for k, v in request.args.items()
                    if k not in keys and k not in ("scope", "page", "mode")}
            return url_for("samples", mode="browse", **kept)
        chips = []
        if bid.isdigit():
            lbl = next((f'run {o["run_id"]} · {o["model_tag"]} · '
                        f'{o["task"]}({o["fewshot_k"]})'
                        for o in ctx["benches"]
                        if str(o["benchmark_id"]) == bid), f"benchmark {bid}")
            chips.append((lbl, url_without("benchmark_id")))
        if run.isdigit() and not bid.isdigit():
            chips.append((f'run {run} · {run_names.get(int(run), "")}',
                          url_without("run")))
        if model and not bid.isdigit():
            chips.append((model, url_without("model")))
        if task and not bid.isdigit():
            chips.append((task, url_without("task")))
        if outcome in ("wrong", "right"):
            chips.append((f"{outcome} only", url_without("outcome")))

        try:
            page = max(1, int(request.args.get("page", "1") or 1))
        except ValueError:
            page = 1
        total = q1(f"SELECT COUNT(*) AS n FROM samples s "
                   f"JOIN benchmarks b USING (benchmark_id) "
                   f"WHERE {' AND '.join(where)}", params)["n"]
        img_col = ("s.image_ids" if has_images() else "NULL AS image_ids")
        ext_col = ("s.extracted" if "extracted" in _columns("samples")
                   else "json_extract(s.responses, '$[0].extracted') AS extracted")
        sr_col = ("s.stop_reason" if "stop_reason" in _columns("samples")
                  else "json_extract(s.responses, '$[0].stop_reason') AS stop_reason")
        rows = q(f"""
            SELECT s.sample_pk, s.sample_id, s.benchmark_id, s.model_tag,
                   b.task, s.gold, {ext_col}, {sr_col}, s.score,
                   s.verified_score, {img_col},
                   substr(s.prompt, 1, 500) AS prompt_snip,
                   substr(json_extract(s.responses, '$[0].text'), 1, 1200)
                       AS response_snip
            FROM samples s JOIN benchmarks b USING (benchmark_id)
            WHERE {' AND '.join(where)}
            ORDER BY s.benchmark_id, CAST(s.sample_id AS INTEGER), s.sample_id
            LIMIT ? OFFSET ?""", params + [PAGE_SIZE, (page - 1) * PAGE_SIZE])
        ctx.update(rows=rows, total=total, page=page, page_size=PAGE_SIZE,
                   scope_groups=groups, scope_tasks=scope_tasks,
                   scope_models=scope_models, scope_value=scope_value,
                   chips=chips)
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
        has_ext = "extracted" in _columns("samples")
        ans_a = "sa.extracted" if has_ext else "json_extract(sa.responses, '$[0].extracted')"
        ans_b = "sb.extracted" if has_ext else "json_extract(sb.responses, '$[0].extracted')"
        raw = q(f"""
            SELECT sa.sample_pk AS pk_a, sb.sample_pk AS pk_b, sa.sample_id,
                   sa.gold, {ans_a} AS ans_a, {ans_b} AS ans_b,
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
        try:
            image_ids = json.loads(row["image_ids"] or "[]")
        except (IndexError, KeyError, ValueError):
            image_ids = []
        return render_template("sample.html", s=row, responses=responses,
                               siblings=siblings, image_ids=image_ids)

    @app.route("/image/<int:image_id>")
    def image(image_id):
        try:
            row = q1("SELECT mime, data FROM images WHERE image_id = ?",
                     [image_id])
        except sqlite3.OperationalError:
            row = None  # no images table yet (old ledger)
        if row is None:
            abort(404)
        from flask import Response
        return Response(row["data"], mimetype=row["mime"] or "image/png",
                        headers={"Cache-Control": "public, max-age=86400"})

    return app


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="lm-eval-ledger serve",
        description="Serve the ledger viewer web app (read-only, except "
                    "the confirm-gated delete/compact actions in Run History).")
    p.add_argument("--db", type=Path, default=Path("results/ledger.sqlite3"))
    p.add_argument("--host", default="127.0.0.1",
                   help="bind address (default localhost-only; use "
                        "0.0.0.0 to expose, ideally behind a reverse proxy)")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--token", default=None,
                   help="shared access token gating all pages (team sharing)")
    p.add_argument("--read-only", action="store_true",
                   help="disable the delete/compact actions entirely "
                        "(for public deployments)")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args(argv)
    if not args.db.exists():
        print(f"[ERROR] Ledger not found: {args.db}", file=sys.stderr)
        sys.exit(1)
    app = create_app(args.db, token=args.token,
                     read_only=args.read_only)
    mode = " (read-only)" if args.read_only else ""
    print(f"[SERVE] Ledger viewer on http://{args.host}:{args.port} "
          f"(db: {args.db}){mode}")
    if args.debug:
        # Flask dev server: auto-reload + debugger, development only
        app.run(host=args.host, port=args.port, debug=True, threaded=True)
    else:
        from waitress import serve as waitress_serve
        waitress_serve(app, host=args.host, port=args.port, threads=8)


def _backends_of(config_yaml: str | None) -> str:
    """'vllm', or 'vllm, server' when per-model overrides mix backends."""
    if not config_yaml:
        return ""
    try:
        import yaml
        cfg = yaml.safe_load(config_yaml) or {}
    except Exception:
        return ""
    seen: list[str] = []
    base = cfg.get("backend")
    for m in cfg.get("models") or []:
        b = m.get("backend", base) if isinstance(m, dict) else base
        if b and b not in seen:
            seen.append(b)
    if base and base not in seen and not cfg.get("models"):
        seen.append(base)
    return ", ".join(seen)
