"""FastAPI application factory.

Exposes the REST surface used by both pod workers and (in Phase 3)
the bash framework tools the parent calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi import Response
from fastapi.responses import JSONResponse

from pydantic import BaseModel

from framework import services as svc
from framework.config import load_config
from framework.db import Database, init_db, utcnow_iso
from framework.events import emit_event, record_parent_action
from framework.models import (
    ArtifactCreate, ArtifactOut, BudgetEntry, EventOut, FailureIn,
    GateRejectIn, PodOut, PodRegister, SubmitResultIn,
    TaskCreate, TaskEdit, TaskOut,
)
from framework.scheduler import budget_cap_hit_today
from framework.state import StatePaths


class ParentActionIn(BaseModel):
    tool: str
    args: dict[str, Any] = {}
    result: str = "ok"
    caller: str = "parent"


class SummaryUpdateIn(BaseModel):
    content: str


class DBQueryIn(BaseModel):
    sql: str
    params: list[Any] = []


def create_app(state_dir: str | Path) -> FastAPI:
    paths = StatePaths(state_dir)
    paths.ensure()
    init_db(paths.db)
    db = Database(paths.db)

    config = load_config(paths.config_yaml if paths.config_yaml.exists() else None)

    app = FastAPI(title="framework-backend", version="0.1.0")
    app.state.db = db
    app.state.paths = paths
    app.state.config = config

    def get_db() -> Database:
        return db

    def get_paths() -> StatePaths:
        return paths

    def get_config() -> dict:
        # Re-read on every request so config edits take effect without a
        # backend restart. Cheap (small YAML).
        return load_config(
            paths.config_yaml if paths.config_yaml.exists() else None
        )

    @app.exception_handler(svc.TaskNotFound)
    async def _not_found(_req, exc):
        return JSONResponse(status_code=404, content={"detail": f"not found: {exc}"})

    @app.exception_handler(svc.IllegalTransition)
    async def _illegal(_req, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    # ---------------- Tasks ------------------------------------------

    @app.post("/tasks", response_model=TaskOut)
    def create_task(
        spec: TaskCreate,
        initial_status: str = Query("before_gate"),
        db: Database = Depends(get_db),
        paths: StatePaths = Depends(get_paths),
    ):
        return svc.create_task(db, paths.events_jsonl, spec,
                               initial_status=initial_status)

    @app.get("/tasks", response_model=list[TaskOut])
    def list_tasks(
        status: str | None = None,
        include_archived: bool = False,
        db: Database = Depends(get_db),
    ):
        return svc.list_tasks(db, status=status, include_archived=include_archived)

    @app.get("/tasks/{task_id}", response_model=TaskOut)
    def get_task(task_id: str, db: Database = Depends(get_db)):
        return svc.get_task(db, task_id)

    @app.patch("/tasks/{task_id}", response_model=TaskOut)
    def edit_task(
        task_id: str,
        edit: TaskEdit,
        db: Database = Depends(get_db),
        paths: StatePaths = Depends(get_paths),
    ):
        return svc.edit_task(db, paths.events_jsonl, task_id, edit)

    # ---------------- Gates ------------------------------------------

    @app.post("/tasks/{task_id}/gate/before/approve", response_model=TaskOut)
    def gate_before_approve(
        task_id: str,
        db: Database = Depends(get_db),
        paths: StatePaths = Depends(get_paths),
    ):
        return svc.approve_before(
            db, paths.events_jsonl, task_id, state_paths=paths,
        )

    @app.post("/tasks/{task_id}/gate/before/reject", response_model=TaskOut)
    def gate_before_reject(
        task_id: str,
        body: GateRejectIn,
        db: Database = Depends(get_db),
        paths: StatePaths = Depends(get_paths),
    ):
        return svc.reject_before(db, paths.events_jsonl, task_id, body.reason)

    @app.post("/tasks/{task_id}/gate/after/approve", response_model=TaskOut)
    def gate_after_approve(
        task_id: str,
        db: Database = Depends(get_db),
        paths: StatePaths = Depends(get_paths),
    ):
        return svc.approve_after(
            db, paths.events_jsonl, task_id, state_paths=paths,
        )

    @app.post("/tasks/{task_id}/gate/after/reject", response_model=TaskOut)
    def gate_after_reject(
        task_id: str,
        body: GateRejectIn,
        db: Database = Depends(get_db),
        paths: StatePaths = Depends(get_paths),
    ):
        return svc.reject_after(
            db, paths.events_jsonl, task_id, body.reason, state_paths=paths,
        )

    # ---------------- Pod operations --------------------------------

    @app.post("/pods", response_model=PodOut)
    def register_pod(
        body: PodRegister,
        db: Database = Depends(get_db),
        paths: StatePaths = Depends(get_paths),
    ):
        return svc.register_pod(db, paths.events_jsonl, body.pod_id)

    @app.get("/pods", response_model=list[PodOut])
    def list_pods(db: Database = Depends(get_db)):
        return svc.list_pods(db)

    @app.post("/pods/{pod_id}/claim")
    def pod_claim(
        pod_id: str,
        db: Database = Depends(get_db),
        paths: StatePaths = Depends(get_paths),
        cfg: dict = Depends(get_config),
    ):
        cap = cfg.get("budget", {}).get("daily_cap_usd")
        task = svc.claim(
            db, paths.events_jsonl, pod_id,
            daily_cap_usd=float(cap) if cap else None,
        )
        if task is None:
            return Response(status_code=204)
        return task.model_dump()

    @app.post("/tasks/{task_id}/start", response_model=TaskOut)
    def task_start(
        task_id: str,
        db: Database = Depends(get_db),
        paths: StatePaths = Depends(get_paths),
    ):
        return svc.mark_running(db, paths.events_jsonl, task_id)

    @app.post("/tasks/{task_id}/submit")
    def task_submit(
        task_id: str,
        body: SubmitResultIn,
        db: Database = Depends(get_db),
        paths: StatePaths = Depends(get_paths),
    ):
        task, artifacts = svc.submit_result(
            db, paths.events_jsonl, paths.budget_ledger_jsonl, task_id, body,
            logs_dir=paths.logs_dir,
        )
        return {
            "task": task.model_dump(),
            "artifacts": [a.model_dump() for a in artifacts],
        }

    @app.post("/tasks/{task_id}/fail", response_model=TaskOut)
    def task_fail(
        task_id: str,
        body: FailureIn,
        db: Database = Depends(get_db),
        paths: StatePaths = Depends(get_paths),
    ):
        return svc.report_failure(
            db, paths.events_jsonl, task_id, body,
            logs_dir=paths.logs_dir,
        )

    @app.post("/tasks/{task_id}/requeue", response_model=TaskOut)
    def task_requeue(
        task_id: str,
        db: Database = Depends(get_db),
        paths: StatePaths = Depends(get_paths),
    ):
        return svc.requeue_task(db, paths.events_jsonl, task_id)

    @app.post("/session/reset")
    def session_reset(
        db: Database = Depends(get_db),
        paths: StatePaths = Depends(get_paths),
    ):
        return svc.session_reset(db, paths.events_jsonl)

    # ---------------- Artifacts -------------------------------------

    @app.get("/artifacts", response_model=list[ArtifactOut])
    def list_artifacts(
        type: str | None = None,
        task_id: str | None = None,
        db: Database = Depends(get_db),
    ):
        return svc.list_artifacts(db, artifact_type=type, task_id=task_id)

    @app.get("/artifacts/{artifact_id}", response_model=ArtifactOut)
    def get_artifact(artifact_id: str, db: Database = Depends(get_db)):
        return svc.get_artifact(db, artifact_id)

    # ---------------- Events / budget -------------------------------

    @app.get("/events", response_model=list[EventOut])
    def list_events(
        limit: int = 100,
        task_id: str | None = None,
        db: Database = Depends(get_db),
    ):
        if task_id:
            rows = db.query_all(
                "SELECT * FROM events WHERE task_id = ? ORDER BY ts ASC LIMIT ?",
                (task_id, limit),
            )
        else:
            rows = db.query_all(
                "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)
            )
        return [
            EventOut(
                event_id=r["event_id"],
                ts=r["ts"],
                type=r["type"],
                task_id=r["task_id"],
                payload=json.loads(r["payload"] or "{}"),
            )
            for r in rows
        ]

    @app.get("/budget", response_model=list[BudgetEntry])
    def list_budget(
        limit: int = 100,
        task_id: str | None = None,
        db: Database = Depends(get_db),
    ):
        if task_id:
            rows = db.query_all(
                "SELECT * FROM budget_ledger WHERE task_id = ? "
                "ORDER BY ts ASC LIMIT ?",
                (task_id, limit),
            )
        else:
            rows = db.query_all(
                "SELECT * FROM budget_ledger ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
        return [BudgetEntry(**dict(r)) for r in rows]

    @app.get("/budget/total")
    def budget_total(db: Database = Depends(get_db)):
        row = db.query_one(
            "SELECT COALESCE(SUM(cost_usd), 0) AS total_usd, "
            "COALESCE(SUM(input_tokens), 0) AS input_tokens, "
            "COALESCE(SUM(output_tokens), 0) AS output_tokens "
            "FROM budget_ledger"
        )
        return dict(row) if row else {"total_usd": 0.0}

    # ---------------- Parent actions log ---------------------------

    @app.post("/parent_actions")
    def post_parent_action(
        body: ParentActionIn,
        db: Database = Depends(get_db),
        paths: StatePaths = Depends(get_paths),
    ):
        record_parent_action(
            db, paths.parent_actions_jsonl,
            tool=body.tool, args=body.args,
            result=body.result, caller=body.caller,
        )
        return {"ok": True}

    # ---------------- Rolling summary ------------------------------

    @app.post("/summary")
    def post_summary(
        body: SummaryUpdateIn,
        db: Database = Depends(get_db),
        paths: StatePaths = Depends(get_paths),
    ):
        paths.rolling_summary.parent.mkdir(parents=True, exist_ok=True)
        paths.rolling_summary.write_text(body.content, encoding="utf-8")
        emit_event(
            db, paths.events_jsonl, "summary_updated",
            payload={"bytes": len(body.content)},
        )
        return {"ok": True, "bytes": len(body.content)}

    @app.get("/summary")
    def get_summary(paths: StatePaths = Depends(get_paths)):
        if not paths.rolling_summary.exists():
            return {"content": ""}
        return {"content": paths.rolling_summary.read_text(encoding="utf-8")}

    # ---------------- State snapshot -------------------------------

    @app.get("/state")
    def get_state(
        recent_events: int = 10,
        db: Database = Depends(get_db),
        cfg: dict = Depends(get_config),
    ):
        """Snapshot for `framework state`: pod status, queue counts,
        recent events, pending gates, budget total."""
        pods = [dict(r) for r in db.query_all("SELECT * FROM pods")]
        counts = {
            r["status"]: r["n"] for r in db.query_all(
                "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status"
            )
        }
        pending_before = [
            TaskOut.from_row(r).model_dump() for r in db.query_all(
                "SELECT * FROM tasks WHERE status = 'before_gate' "
                "ORDER BY priority DESC, created_at ASC"
            )
        ]
        pending_after = [
            TaskOut.from_row(r).model_dump() for r in db.query_all(
                "SELECT * FROM tasks WHERE status = 'after_gate' "
                "ORDER BY completed_at ASC"
            )
        ]
        events = [
            {
                "event_id": r["event_id"],
                "ts": r["ts"],
                "type": r["type"],
                "task_id": r["task_id"],
                "payload": json.loads(r["payload"] or "{}"),
            }
            for r in db.query_all(
                "SELECT * FROM events ORDER BY ts DESC LIMIT ?",
                (recent_events,),
            )
        ]
        budget_row = db.query_one(
            "SELECT COALESCE(SUM(cost_usd), 0) AS total_usd, "
            "COALESCE(SUM(input_tokens), 0) AS input_tokens, "
            "COALESCE(SUM(output_tokens), 0) AS output_tokens "
            "FROM budget_ledger"
        )
        today_row = db.query_one(
            "SELECT COALESCE(SUM(cost_usd), 0) AS total_usd "
            "FROM budget_ledger WHERE substr(ts, 1, 10) = substr(?, 1, 10)",
            (utcnow_iso(),),
        )
        cap = cfg.get("budget", {}).get("daily_cap_usd")
        cap_hit = budget_cap_hit_today(db)
        return {
            "pods": pods,
            "queue_counts": counts,
            "pending_before_gate": pending_before,
            "pending_after_gate": pending_after,
            "recent_events": events,
            "budget": dict(budget_row) if budget_row else {"total_usd": 0.0},
            "budget_today": {
                "spent_usd": round(float(today_row["total_usd"]), 6) if today_row else 0.0,
                "cap_usd": float(cap) if cap else None,
                "cap_hit_today": cap_hit,
            },
        }

    # ---------------- Agent role configs ---------------------------

    @app.get("/agents/{role}")
    def get_agent_config(role: str, paths: StatePaths = Depends(get_paths)):
        p = paths.agents_dir / f"{role}.md"
        if not p.exists():
            raise HTTPException(
                status_code=404,
                detail=f"agent config not found: agents/{role}.md",
            )
        return {"role": role, "content": p.read_text(encoding="utf-8")}

    # ---------------- DB read-only query ---------------------------

    @app.post("/db/query")
    def db_query(body: DBQueryIn, db: Database = Depends(get_db)):
        sql = body.sql.strip().rstrip(";")
        # Strict read-only guard. The methodology's `framework db query`
        # is a parent-side inspection tool; never let it mutate.
        first = sql.split(None, 1)[0].lower() if sql else ""
        if first not in ("select", "with", "pragma", "explain"):
            raise HTTPException(
                status_code=400,
                detail=f"only SELECT/WITH/PRAGMA/EXPLAIN allowed; got: {first!r}",
            )
        try:
            rows = db.query_all(sql, tuple(body.params))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"rows": [dict(r) for r in rows], "row_count": len(rows)}

    # ---------------- Health ----------------------------------------

    @app.get("/health")
    def health():
        return {"ok": True}

    return app
