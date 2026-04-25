"""End-to-end worktree integration: bootstrap a git target, approve a dev
task, observe the worktree on disk + the task row's working_dir/worktree_path."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from framework import services as svc
from framework.api.app import create_app
from framework.bootstrap import bootstrap_run
from framework.cli import commands as C
from framework.cli._context import CliContext
from framework.models import TaskCreate
from framework.pod.backend_client import BackendClient


@pytest.fixture
def gitenv(tmp_path):
    target = tmp_path / "repo"
    target.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=target, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=target, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=target, check=True)
    (target / "README.md").write_text("hi\n")
    subprocess.run(["git", "add", "."], cwd=target, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=target, check=True)

    state_root = tmp_path / "fw"
    info = bootstrap_run(state_root, goal="g", target_repo=str(target))

    app = create_app(state_root)
    test_client = TestClient(app)
    backend = BackendClient(http_client=test_client)
    paths = app.state.paths
    db = app.state.db
    ctx = CliContext(backend=backend, paths=paths)
    yield {
        "ctx": ctx, "db": db, "paths": paths, "target": target,
        "branch": info["branch_name"],
    }
    test_client.close()


def test_dev_task_gets_worktree_on_before_approve(gitenv):
    """Approving a development task creates a worktree, overrides
    working_dir, and stores worktree_path."""
    ctx = gitenv["ctx"]
    target = gitenv["target"]
    branch = gitenv["branch"]

    spec = TaskCreate(
        agent_role="development",
        goal_text="add fizzbuzz.py",
        output_artifact_types=["PatchSummary"],
        working_dir=str(target),  # original target — will be overridden
    )
    t = ctx.backend.create_task(spec.model_dump(), initial_status="before_gate")
    tid = t["task_id"]

    C.cmd_gate_before_approve(ctx, tid)
    after = ctx.backend.get_task(tid)

    # working_dir was overridden to a worktree path
    assert after["working_dir"] != str(target)
    assert after["working_dir"] == after["worktree_path"]
    assert Path(after["worktree_path"]).is_dir()
    assert Path(after["worktree_path"]).name == tid

    # worktree is on the per-task branch
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=after["worktree_path"], capture_output=True, text=True, check=True,
    )
    assert proc.stdout.strip() == f"{branch}-{tid}"


def test_testing_task_gets_its_own_worktree_from_base(gitenv):
    """Testing tasks get a fresh worktree from the base branch on
    approve. By then any dev dependencies have already been merged
    into base (at their own after-gate approve), so the testing
    worktree sees the latest approved state. Read-only access is
    enforced by the role's allowed_tools (no filesystem_write), not
    by branch isolation."""
    ctx = gitenv["ctx"]
    target = gitenv["target"]

    test_spec = TaskCreate(
        agent_role="testing", goal_text="run tests",
        output_artifact_types=["TestResult"], working_dir=str(target),
    )
    test = ctx.backend.create_task(test_spec.model_dump(),
                                   initial_status="before_gate")
    C.cmd_gate_before_approve(ctx, test["task_id"])
    after = ctx.backend.get_task(test["task_id"])

    assert after["worktree_path"] is not None
    assert after["working_dir"] == after["worktree_path"]
    assert Path(after["worktree_path"]).is_dir()
    assert Path(after["worktree_path"]).name == test["task_id"]


def _force_claim(db, task_id: str, pod_id: str = "pod_a") -> None:
    """Bypass the scheduler and put a specific task into 'claimed' so
    we can submit_result against it without worrying about claim order."""
    from framework.db import utcnow_iso
    now = utcnow_iso()
    db.execute(
        "UPDATE tasks SET status='claimed', pod_id=?, claimed_at=? WHERE task_id=?",
        (pod_id, now, task_id),
    )


def _submit_canned_patch(ctx, db, task_id: str, *, content=None,
                         artifact_type: str = "PatchSummary",
                         agent: str = "development"):
    from framework.models import ArtifactCreate, SubmitResultIn
    _force_claim(db, task_id)
    artifact = ArtifactCreate(
        artifact_type=artifact_type,
        produced_by_task=task_id,
        produced_by_agent=agent,
        content=content or {"files_changed": ["scratch.py"], "rationale": "ok",
                            "test_targets": [], "diff_stat": {}},
    )
    result = SubmitResultIn(
        artifacts=[artifact], input_tokens=10, output_tokens=2,
        cost_usd=0.001, duration_seconds=0.1, model="claude-haiku-4-5-20251001",
    )
    ctx.backend.submit_result(task_id, result.model_dump())


def test_after_gate_approve_captures_diff_and_removes_worktree(gitenv):
    """Dev task approves → diff captured into PatchSummary, worktree gone."""
    from framework.models import TaskCreate
    ctx, target = gitenv["ctx"], gitenv["target"]
    ctx.backend.register_pod("pod_a")

    spec = TaskCreate(
        agent_role="development", goal_text="add fizz.py",
        output_artifact_types=["PatchSummary"], working_dir=str(target),
    )
    t = ctx.backend.create_task(spec.model_dump(), initial_status="before_gate")
    tid = t["task_id"]
    C.cmd_gate_before_approve(ctx, tid)
    after_approve = ctx.backend.get_task(tid)
    wt_path = Path(after_approve["worktree_path"])

    # Simulate the pod doing work in the worktree.
    (wt_path / "fizz.py").write_text("FIZZ = 1\n")

    # Submit a canned artifact to drive the task to after_gate.
    _submit_canned_patch(ctx, gitenv["db"], tid)

    C.cmd_gate_after_approve(ctx, tid)

    # Worktree gone, worktree_path cleared.
    final = ctx.backend.get_task(tid)
    assert final["status"] == "done"
    assert final["worktree_path"] is None
    assert not wt_path.exists()

    # Diff captured into the PatchSummary.
    arts = ctx.backend.list_artifacts(task_id=tid)
    patch = next(a for a in arts if a["artifact_type"] == "PatchSummary")
    assert "diff" in patch["content"]
    assert "fizz.py" in patch["content"]["diff"]


def test_after_gate_reject_removes_worktree_without_diff(gitenv):
    """Reject path: worktree cleaned up, no diff captured (the user
    already saw the artifact and rejected it)."""
    from framework.models import TaskCreate
    ctx, target = gitenv["ctx"], gitenv["target"]
    ctx.backend.register_pod("pod_a")

    spec = TaskCreate(
        agent_role="development", goal_text="x",
        output_artifact_types=["PatchSummary"], working_dir=str(target),
    )
    t = ctx.backend.create_task(spec.model_dump(), initial_status="before_gate")
    tid = t["task_id"]
    C.cmd_gate_before_approve(ctx, tid)
    wt_path = Path(ctx.backend.get_task(tid)["worktree_path"])
    (wt_path / "garbage.py").write_text("oops\n")
    _submit_canned_patch(ctx, gitenv["db"], tid)

    C.cmd_gate_after_reject(ctx, tid, "wrong file")

    final = ctx.backend.get_task(tid)
    # Reject sends the task back to before_gate with retry_count bumped.
    assert final["status"] == "before_gate"
    assert final["retry_count"] == 1
    assert final["worktree_path"] is None
    assert not wt_path.exists()

    # No diff was attached to the artifact.
    arts = ctx.backend.list_artifacts(task_id=tid)
    patch = next(a for a in arts if a["artifact_type"] == "PatchSummary")
    assert "diff" not in patch["content"]


def test_testing_approve_removes_own_worktree_without_merging(gitenv):
    """Testing tasks own their worktree (since v2.1) and clean it up
    on after-gate approve. But unlike dev: no auto-commit, no merge —
    the testing role is read-only by contract."""
    from framework.models import TaskCreate
    ctx, target = gitenv["ctx"], gitenv["target"]
    ctx.backend.register_pod("pod_a")

    test_spec = TaskCreate(
        agent_role="testing", goal_text="run tests",
        output_artifact_types=["TestResult"], working_dir=str(target),
    )
    test = ctx.backend.create_task(test_spec.model_dump(),
                                   initial_status="before_gate")
    C.cmd_gate_before_approve(ctx, test["task_id"])
    test_wt = Path(ctx.backend.get_task(test["task_id"])["worktree_path"])
    assert test_wt.exists()

    _submit_canned_patch(
        ctx, gitenv["db"], test["task_id"],
        artifact_type="TestResult", agent="testing",
        content={"tests_run": 1, "passed": 1, "failed": [],
                 "runtime_seconds": 0.1},
    )
    C.cmd_gate_after_approve(ctx, test["task_id"])

    # Worktree gone; worktree_path cleared.
    final = ctx.backend.get_task(test["task_id"])
    assert final["status"] == "done"
    assert final["worktree_path"] is None
    assert not test_wt.exists()
    # TestResult artifact has no `diff` field — testing role doesn't
    # auto-commit or capture diffs.
    arts = ctx.backend.list_artifacts(task_id=test["task_id"])
    tr = next(a for a in arts if a["artifact_type"] == "TestResult")
    assert "diff" not in tr["content"]


def test_worktree_skipped_for_non_git_target(tmp_path):
    """Non-git targets bootstrap fine and approve_before is a no-op
    for worktrees — pods just edit the original working_dir."""
    target = tmp_path / "plain"
    target.mkdir()
    state_root = tmp_path / "fw"
    bootstrap_run(state_root, goal="g", target_repo=str(target))

    app = create_app(state_root)
    test_client = TestClient(app)
    try:
        backend = BackendClient(http_client=test_client)
        ctx = CliContext(backend=backend, paths=app.state.paths)
        spec = TaskCreate(
            agent_role="development", goal_text="x",
            output_artifact_types=["PatchSummary"], working_dir=str(target),
        )
        t = ctx.backend.create_task(spec.model_dump(),
                                    initial_status="before_gate")
        C.cmd_gate_before_approve(ctx, t["task_id"])
        after = ctx.backend.get_task(t["task_id"])
        assert after["working_dir"] == str(target.resolve()) or \
               after["working_dir"] == str(target)  # untouched
        assert after["worktree_path"] is None
    finally:
        test_client.close()
