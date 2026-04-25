"""Phase 3: framework tools end-to-end against a TestClient backend.

Each command path is exercised, and parent_actions logging is verified
on the side that matters most (it's the audit trail).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from framework import services as svc
from framework.api.app import create_app
from framework.cli import commands as C
from framework.cli._context import CliContext
from framework.models import (
    ArtifactCreate, SubmitResultIn, TaskCreate,
)
from framework.pod.backend_client import BackendClient


@pytest.fixture
def cli_env(tmp_path):
    from tests.conftest import _copy_agent_templates
    state_root = tmp_path / "fw"
    app = create_app(state_root)
    _copy_agent_templates(app.state.paths)
    test_client = TestClient(app)
    backend = BackendClient(http_client=test_client)
    paths = app.state.paths
    db = app.state.db
    out = io.StringIO()
    err = io.StringIO()
    ctx = CliContext(backend=backend, paths=paths, stdout=out, stderr=err)
    yield ctx, db, paths, out, err
    test_client.close()


def _seed_task(db, paths, **overrides) -> str:
    spec = TaskCreate(
        agent_role=overrides.pop("agent_role", "development"),
        goal_text=overrides.pop("goal_text", "do thing"),
        recommended_model=overrides.pop("model", "claude-haiku-4-5-20251001"),
        priority=overrides.pop("priority", 0),
        output_artifact_types=overrides.pop("output_artifact_types", ["PatchSummary"]),
    )
    return svc.create_task(db, paths.events_jsonl, spec).task_id


def _parent_actions(db) -> list[dict]:
    return [dict(r) for r in db.query_all(
        "SELECT * FROM parent_actions ORDER BY id ASC"
    )]


# ---------------- inspection -----------------------------------------

def test_state_dump_and_logging(cli_env):
    ctx, db, paths, out, _ = cli_env
    _seed_task(db, paths)
    rc = C.cmd_state(ctx)
    assert rc == 0
    blob = yaml.safe_load(out.getvalue())
    assert "queue_counts" in blob
    assert blob["queue_counts"].get("before_gate") == 1
    assert isinstance(blob["pending_before_gate"], list)
    actions = _parent_actions(db)
    assert any(a["tool"] == "framework_state" for a in actions)
    # JSONL mirror
    assert paths.parent_actions_jsonl.exists()
    line = paths.parent_actions_jsonl.read_text().splitlines()[0]
    assert json.loads(line)["tool"] == "framework_state"


def test_db_query_select_only(cli_env):
    ctx, db, paths, out, _ = cli_env
    _seed_task(db, paths)
    rc = C.cmd_db_query(ctx, "SELECT count(*) AS n FROM tasks")
    assert rc == 0
    parsed = yaml.safe_load(out.getvalue())
    assert parsed["row_count"] == 1
    assert parsed["rows"][0]["n"] == 1


def test_db_query_rejects_mutations(cli_env):
    import httpx
    ctx, _db, _paths, _, _ = cli_env
    with pytest.raises(httpx.HTTPStatusError):
        C.cmd_db_query(ctx, "DELETE FROM tasks")


def test_artifact_get_and_list(cli_env):
    ctx, db, paths, out, _ = cli_env
    tid = _seed_task(db, paths)
    svc.approve_before(db, paths.events_jsonl, tid)
    svc.register_pod(db, paths.events_jsonl, "pod_a")
    svc.claim(db, paths.events_jsonl, "pod_a")
    task, arts = svc.submit_result(
        db, paths.events_jsonl, paths.budget_ledger_jsonl, tid,
        SubmitResultIn(artifacts=[ArtifactCreate(
            artifact_type="PatchSummary",
            produced_by_task=tid, produced_by_agent="development",
            model="claude-haiku-4-5-20251001",
            content={"files_changed": ["a.py"], "rationale": "x"},
        )], input_tokens=10, output_tokens=5, cost_usd=0.0001,
        duration_seconds=0.5, model="claude-haiku-4-5-20251001"),
    )
    artifact_id = arts[0].artifact_id

    # list
    out.truncate(0); out.seek(0)
    C.cmd_artifact_list(ctx, type="PatchSummary")
    listed = yaml.safe_load(out.getvalue())
    assert len(listed) == 1
    assert listed[0]["artifact_id"] == artifact_id

    # get
    out.truncate(0); out.seek(0)
    C.cmd_artifact_get(ctx, artifact_id)
    fetched = yaml.safe_load(out.getvalue())
    assert fetched["content"] == {"files_changed": ["a.py"], "rationale": "x"}
    assert fetched["model"] == "claude-haiku-4-5-20251001"

    actions = _parent_actions(db)
    assert any(a["tool"] == "framework_artifact_list" for a in actions)
    assert any(a["tool"] == "framework_artifact_get" for a in actions)


# ---------------- plan create / edit ---------------------------------

def test_plan_create_lands_in_before_gate(cli_env, tmp_path):
    ctx, db, paths, out, _ = cli_env
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.write_text(yaml.safe_dump({
        "tasks": [
            {
                "agent_role": "development",
                "goal_text": "Add UCI handshake",
                "recommended_model": "claude-sonnet-4-6",
                "output_artifact_types": ["PatchSummary"],
                "priority": 5,
            },
            {
                "agent_role": "testing",
                "goal_text": "Run integration tests",
                "recommended_model": "claude-haiku-4-5-20251001",
                "output_artifact_types": ["TestResult"],
                "priority": 2,
            },
        ],
    }))
    rc = C.cmd_plan_create(ctx, plan_yaml)
    assert rc == 0
    tasks = svc.list_tasks(db)
    assert len(tasks) == 2
    assert all(t.status == "before_gate" for t in tasks)
    actions = _parent_actions(db)
    assert any(a["tool"] == "framework_plan_create" for a in actions)


def test_plan_edit_parses_lists(cli_env):
    ctx, db, paths, out, _ = cli_env
    tid = _seed_task(db, paths)
    rc = C.cmd_plan_edit(ctx, tid, "depends_on", "['t_x', 't_y']")
    assert rc == 0
    t = svc.get_task(db, tid)
    assert t.depends_on == ["t_x", "t_y"]


def test_plan_edit_parses_int(cli_env):
    ctx, db, paths, _, _ = cli_env
    tid = _seed_task(db, paths)
    C.cmd_plan_edit(ctx, tid, "priority", "9")
    assert svc.get_task(db, tid).priority == 9


# ---------------- gate transitions ----------------------------------

def test_gate_before_approve_and_reject(cli_env):
    ctx, db, paths, _, _ = cli_env
    tid_a = _seed_task(db, paths)
    tid_r = _seed_task(db, paths)

    C.cmd_gate_before_approve(ctx, tid_a)
    assert svc.get_task(db, tid_a).status == "ready"

    C.cmd_gate_before_reject(ctx, tid_r, reason="not now")
    t = svc.get_task(db, tid_r)
    assert t.status == "rejected"
    assert t.rejection_reason == "not now"

    actions = _parent_actions(db)
    tools = [a["tool"] for a in actions]
    assert "framework_gate_before_approve" in tools
    assert "framework_gate_before_reject" in tools


def test_gate_before_approve_batch(cli_env):
    """Approve multiple tasks in one CLI invocation. The motivation:
    when two independent tasks have no dependencies on each other,
    flipping them to 'ready' in the same process collapses ~1–2s of
    Python boot overhead per task into one — letting two pods
    polling at 2s claim them in the same window."""
    ctx, db, paths, _, _ = cli_env
    t1 = _seed_task(db, paths)
    t2 = _seed_task(db, paths)
    t3 = _seed_task(db, paths)

    rc = C.cmd_gate_before_approve(ctx, [t1, t2, t3])
    assert rc == 0
    assert svc.get_task(db, t1).status == "ready"
    assert svc.get_task(db, t2).status == "ready"
    assert svc.get_task(db, t3).status == "ready"

    # parent_actions has one row per approved task — auditable.
    actions = [a for a in _parent_actions(db)
               if a["tool"] == "framework_gate_before_approve"]
    approved = {json.loads(a["args"])["task_id"] for a in actions}
    assert approved == {t1, t2, t3}


def test_gate_before_approve_argparse_passes_list_of_ids():
    """The CLI parser binds nargs='+' so the command always receives
    a list, even for a single ID. Lock that contract in."""
    from framework.cli.parser import build_parser
    args = build_parser().parse_args(
        ["gate", "before", "approve", "t_a", "t_b"]
    )
    assert args.task_id == ["t_a", "t_b"]
    args1 = build_parser().parse_args(["gate", "before", "approve", "t_solo"])
    assert args1.task_id == ["t_solo"]


def test_gate_after_approve_and_reject_cycle(cli_env):
    ctx, db, paths, out, _ = cli_env
    tid = _seed_task(db, paths)
    svc.approve_before(db, paths.events_jsonl, tid)
    svc.register_pod(db, paths.events_jsonl, "pod_a")
    svc.claim(db, paths.events_jsonl, "pod_a")
    svc.submit_result(
        db, paths.events_jsonl, paths.budget_ledger_jsonl, tid,
        SubmitResultIn(artifacts=[ArtifactCreate(
            artifact_type="PatchSummary",
            produced_by_task=tid, produced_by_agent="development",
            content={"x": 1},
        )]),
    )
    # reject path → returns to before_gate
    C.cmd_gate_after_reject(ctx, tid, reason="redo")
    t = svc.get_task(db, tid)
    assert t.status == "before_gate"
    assert t.retry_count == 1

    # approve path
    svc.approve_before(db, paths.events_jsonl, tid)
    svc.register_pod(db, paths.events_jsonl, "pod_a")
    svc.claim(db, paths.events_jsonl, "pod_a")
    svc.submit_result(
        db, paths.events_jsonl, paths.budget_ledger_jsonl, tid,
        SubmitResultIn(artifacts=[ArtifactCreate(
            artifact_type="PatchSummary",
            produced_by_task=tid, produced_by_agent="development",
            content={"x": 2},
        )]),
    )
    out.truncate(0); out.seek(0)
    C.cmd_gate_after_approve(ctx, tid)
    assert svc.get_task(db, tid).status == "done"
    # The after-approve message reminds the parent about the rolling summary
    assert "rolling_summary" in out.getvalue() or "summary update" in out.getvalue()


# ---------------- summary -------------------------------------------

def test_summary_update_writes_file_and_emits_event(cli_env, tmp_path):
    ctx, db, paths, _, _ = cli_env
    summary_file = tmp_path / "new_summary.md"
    summary_file.write_text("## Goal\nx\n## Completed milestones\n- y\n")

    rc = C.cmd_summary_update(ctx, summary_file)
    assert rc == 0
    assert paths.rolling_summary.read_text() == summary_file.read_text()

    types = [r["type"] for r in db.query_all(
        "SELECT type FROM events ORDER BY ts DESC LIMIT 5"
    )]
    assert "summary_updated" in types

    actions = _parent_actions(db)
    assert any(a["tool"] == "framework_summary_update" for a in actions)


def test_summary_warns_when_too_long(cli_env, tmp_path):
    ctx, _, _, _, err = cli_env
    summary = tmp_path / "huge.md"
    summary.write_text(" ".join(["word"] * 3000))
    C.cmd_summary_update(ctx, summary)
    assert "warning" in err.getvalue().lower()


# ---------------- run start -----------------------------------------

def test_run_start_bootstraps(tmp_path):
    """run start operates on a fresh state dir; the cli_env fixture's
    backend points at an already-initialized DB, so use a fresh one."""
    out = io.StringIO()
    err = io.StringIO()

    target = tmp_path / "repo"
    target.mkdir()
    fresh_state = tmp_path / "fresh"

    # Build app on a temp dir; we'll point ctx.paths at the same dir.
    app = create_app(fresh_state)  # initializes the dir up front
    test_client = TestClient(app)
    try:
        backend = BackendClient(http_client=test_client)
        from framework.state import StatePaths
        ctx = CliContext(backend=backend, paths=StatePaths(fresh_state),
                         stdout=out, stderr=err)
        rc = C.cmd_run_start(ctx, goal="Test goal",
                             target_repo=str(target), overwrite=True)
        assert rc == 0
        info = yaml.safe_load(out.getvalue())
        assert info["goal"] == "Test goal"
        assert (fresh_state / "CLAUDE.md").exists()
        assert (fresh_state / "agents" / "methodology.md").exists()
    finally:
        test_client.close()
