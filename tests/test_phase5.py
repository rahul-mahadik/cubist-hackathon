"""Phase 5: agent .md drives the pod prompt; per-agent change logs;
contract enforcement; small-project end-to-end.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from framework import services as svc
from framework.api.app import create_app
from framework.bootstrap import bootstrap_run
from framework.cli import commands as C
from framework.cli._context import CliContext
from framework.cli.subagent import cmd_subagent_invoke
from framework.config import load_config
from framework.pod.backend_client import BackendClient
from framework.pod.prompt import (
    build_pod_prompt, contract_types, parse_agent_md,
)
from framework.pod.worker import process_one_task


# ---------------- fakes ----------------------------------------------

@dataclass
class _Usage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class _Block:
    type: str
    text: str


@dataclass
class _Resp:
    content: list
    usage: _Usage
    stop_reason: str = "end_turn"


class FakeAnthropic:
    def __init__(self, response_text: str, usage: _Usage | None = None,
                 raise_exc: BaseException | None = None):
        self._text = response_text
        self._usage = usage or _Usage()
        self._raise = raise_exc
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise:
            raise self._raise
        return _Resp(content=[_Block(type="text", text=self._text)],
                     usage=self._usage)


def _caller_for(fake: FakeAnthropic):
    from framework.pod.anthropic_call import call_messages, call_messages_agentic
    def caller(**kw):
        if "tools" in kw:
            return call_messages_agentic(fake, **kw)
        return call_messages(fake, **kw)
    return caller


# ---------------- pure prompt tests ----------------------------------

def test_pod_prompt_uses_agent_md_body_as_system():
    md = (
        "---\nrole: development\ndefault_model: claude-sonnet-4-6\n"
        "output_artifact_contract: [PatchSummary, ProgressLogEntry]\n"
        "---\n"
        "# Development Subagent\n\nYou edit code in working_dir.\n"
    )
    task = {
        "task_id": "t_1", "agent_role": "development",
        "goal_text": "Add greet()", "working_dir": "/repo",
        "output_artifact_types": ["PatchSummary"],
    }
    system, user, fm = build_pod_prompt(
        agent_md_text=md, task=task,
        rolling_summary="## Goal\nAdd greet()\n",
        input_artifacts=[{
            "artifact_id": "a_1", "artifact_type": "ResearchBrief",
            "content": {"summary": "spec lives in docs/"},
        }],
    )
    assert "Development Subagent" in system
    assert "edit code in working_dir" in system

    assert "Add greet()" in user
    assert "/repo" in user
    assert "## Goal" in user
    assert "a_1 (ResearchBrief)" in user
    assert "spec lives in docs/" in user
    # JSON shape hint included
    assert "files_changed" in user

    assert fm["default_model"] == "claude-sonnet-4-6"
    assert fm["output_artifact_contract"] == ["PatchSummary", "ProgressLogEntry"]


def test_testresult_prompt_allows_metrics_and_artifact_files():
    md = (
        "---\nrole: testing\ndefault_model: claude-haiku-4-5-20251001\n"
        "output_artifact_contract: [TestResult, ProgressLogEntry]\n"
        "---\n"
        "# Testing Subagent\n\nRun tests and evals.\n"
    )
    task = {
        "task_id": "t_eval", "agent_role": "testing",
        "goal_text": "Run chess eval and report Elo.",
        "working_dir": "/repo",
        "output_artifact_types": ["TestResult"],
    }
    _, user, _ = build_pod_prompt(agent_md_text=md, task=task)
    assert "metrics" in user
    assert "artifact_files" in user


def test_contract_types_strips_progresslog():
    fm, _ = parse_agent_md(
        "---\noutput_artifact_contract: [PatchSummary, ProgressLogEntry]\n---\nbody\n"
    )
    assert contract_types(fm) == ["PatchSummary", "ProgressLogEntry"]


# ---------------- pod worker uses agent .md --------------------------

@pytest.fixture
def env(tmp_path):
    """Bootstrapped framework-state with a TestClient backend."""
    target = tmp_path / "repo"
    target.mkdir()
    (target / "main.py").write_text("def main(): pass\n")
    state_root = tmp_path / "fw"
    bootstrap_run(state_root, goal="Add greet() function",
                  target_repo=str(target))
    app = create_app(state_root)
    test_client = TestClient(app)
    backend = BackendClient(http_client=test_client)
    paths = app.state.paths
    db = app.state.db
    out = io.StringIO()
    err = io.StringIO()
    ctx = CliContext(backend=backend, paths=paths, stdout=out, stderr=err)
    yield ctx, db, paths, out, err, target
    test_client.close()


def _seed_ready_task(db, paths, **overrides):
    from framework.models import TaskCreate
    spec = TaskCreate(
        agent_role=overrides.pop("agent_role", "development"),
        goal_text=overrides.pop("goal_text", "Add greet()"),
        recommended_model=overrides.pop("model", None),
        priority=overrides.pop("priority", 0),
        output_artifact_types=overrides.pop(
            "output_artifact_types", ["PatchSummary"],
        ),
        working_dir=overrides.pop("working_dir", None),
    )
    t = svc.create_task(db, paths.events_jsonl, spec)
    svc.approve_before(db, paths.events_jsonl, t.task_id)
    return t.task_id


def test_pod_calls_with_agent_md_system_prompt(env):
    ctx, db, paths, _, _, target = env
    tid = _seed_ready_task(db, paths, working_dir=str(target))
    cfg = load_config(paths.config_yaml)

    fake = FakeAnthropic(
        '{"files_changed": ["main.py"], "rationale": "added greet()"}'
    )
    status = process_one_task(
        "pod_a", backend=ctx.backend,
        anthropic_caller=_caller_for(fake), config=cfg,
    )
    assert status == "done"
    # The call's system block came from agents/development.md, not the
    # hardcoded Phase 2 string.
    sys_block = fake.calls[0]["system"][0]
    assert sys_block["cache_control"] == {"type": "ephemeral"}
    assert "Development Subagent" in sys_block["text"]
    # The user message includes the working_dir and the rolling summary
    # (which the bootstrap seeded with the goal).
    user_msg = fake.calls[0]["messages"][0]["content"]
    assert str(target) in user_msg
    assert "Add greet() function" in user_msg  # from rolling_summary


def test_pod_uses_frontmatter_default_model_when_task_unset(env):
    """With no recommended_model on the task, the pod should fall back to
    the frontmatter default (sonnet for development), not the global
    sonnet alias."""
    ctx, db, paths, _, _, _ = env
    tid = _seed_ready_task(db, paths, model=None)
    cfg = load_config(paths.config_yaml)
    fake = FakeAnthropic('{"files_changed": [], "rationale": "noop"}')
    process_one_task(
        "pod_a", backend=ctx.backend,
        anthropic_caller=_caller_for(fake), config=cfg,
    )
    assert fake.calls[0]["model"] == "claude-sonnet-4-6"


def test_progress_log_appended_per_agent(env):
    ctx, db, paths, _, _, target = env
    tid = _seed_ready_task(db, paths, working_dir=str(target))
    cfg = load_config(paths.config_yaml)
    fake = FakeAnthropic(
        '{"files_changed": ["main.py", "tests/test_main.py"], '
        '"rationale": "implemented greet() and tested"}',
        usage=_Usage(input_tokens=200, output_tokens=80),
    )
    process_one_task(
        "pod_a", backend=ctx.backend,
        anthropic_caller=_caller_for(fake), config=cfg,
    )
    log_path = paths.logs_dir / "development_agent.md"
    assert log_path.exists()
    content = log_path.read_text()
    assert "# development agent" in content
    lines = [l for l in content.splitlines() if "|" in l]
    assert len(lines) == 1
    line = lines[0]
    # Format: ts | task_id | outcome | tokens_in/out | cost | files | artifact_ids
    parts = [p.strip() for p in line.split("|")]
    assert parts[1] == tid
    assert "implemented greet() and tested" in parts[2]
    assert parts[3] == "200/80"
    assert parts[4].startswith("$")
    assert "main.py" in parts[5]
    assert "tests/test_main.py" in parts[5]
    assert parts[6].startswith("a_")  # artifact id


def test_progress_log_records_failure_too(env):
    ctx, db, paths, _, _, target = env
    tid = _seed_ready_task(db, paths, working_dir=str(target))
    cfg = load_config(paths.config_yaml)
    fake = FakeAnthropic("(unused)", raise_exc=RuntimeError("simulated 500"))
    status = process_one_task(
        "pod_a", backend=ctx.backend,
        anthropic_caller=_caller_for(fake), config=cfg,
    )
    assert status == "failed"
    log = (paths.logs_dir / "development_agent.md").read_text()
    assert "FAILED:" in log
    assert "simulated 500" in log


def test_contract_violation_produces_failure_report(env):
    """A development task that asks for a TestResult violates the
    development contract — the pod should refuse without burning tokens."""
    ctx, db, paths, _, _, target = env
    tid = _seed_ready_task(
        db, paths, working_dir=str(target),
        output_artifact_types=["TestResult"],
    )
    cfg = load_config(paths.config_yaml)
    fake = FakeAnthropic("(unused)")
    status = process_one_task(
        "pod_a", backend=ctx.backend,
        anthropic_caller=_caller_for(fake), config=cfg,
    )
    assert status == "failed"
    # No API call made — contract was checked before that.
    assert fake.calls == []
    # FailureReport is the artifact, task lands at after_gate.
    arts = svc.list_artifacts(db, task_id=tid)
    assert len(arts) == 1
    assert arts[0].artifact_type == "FailureReport"
    assert "contract" in arts[0].content["error_message"].lower()


def test_testing_role_uses_testing_agent_md(env):
    ctx, db, paths, _, _, target = env
    tid = _seed_ready_task(
        db, paths, agent_role="testing",
        goal_text="Run tests on greet()",
        output_artifact_types=["TestResult"],
        working_dir=str(target),
    )
    cfg = load_config(paths.config_yaml)
    fake = FakeAnthropic(
        '{"tests_run": 5, "passed": 4, "failed": [{"name": "t_x", "brief_reason": "off-by-one"}], "runtime_seconds": 0.3}'
    )
    process_one_task(
        "pod_a", backend=ctx.backend,
        anthropic_caller=_caller_for(fake), config=cfg,
    )
    assert "Testing Subagent" in fake.calls[0]["system"][0]["text"]
    log = (paths.logs_dir / "testing_agent.md").read_text()
    # outcome line synthesized from TestResult content
    assert "4 passed" in log and "1 failed" in log


# ---------------- end-to-end on a small mock project ----------------

def test_small_project_end_to_end(env):
    """Goal → methodology → 2-task plan (dev + testing) → both gates per
    task → both pods process → both gates → final state. Verifies:

    - methodology agent .md drives the planning prompt
    - both per-agent change logs exist with one entry each
    - rolling summary update reflects in disk + event stream
    - parent_actions covers every framework tool used
    - budget ledger records both pod calls
    """
    ctx, db, paths, out, _, target = env

    # 1. Methodology subagent invocation
    task_yaml = paths.root / "task.yaml"
    task_yaml.write_text(yaml.safe_dump({
        "goal": "Add greet() function with tests",
        "target_repo": str(target),
    }))
    methodology_fake = FakeAnthropic(json.dumps({
        "rationale": "implement then test",
        "tasks": [
            {
                "agent_role": "development",
                "goal_text": "Implement greet() in main.py",
                "recommended_model": "claude-haiku-4-5-20251001",
                "output_artifact_types": ["PatchSummary"],
                "depends_on": [], "priority": 5,
                "rationale": "the implementation",
            },
            {
                "agent_role": "testing",
                "goal_text": "Run unit tests for greet() and report results",
                "recommended_model": "claude-haiku-4-5-20251001",
                "output_artifact_types": ["TestResult"],
                "depends_on": [0], "priority": 3,
                "rationale": "verify the implementation",
            },
        ],
    }))
    cmd_subagent_invoke(
        ctx, "methodology", str(task_yaml),
        anthropic_caller=_caller_for(methodology_fake),
    )
    plan_out = yaml.safe_load(out.getvalue())
    proposed = plan_out["proposed_plan_path"]

    # 2. Plan create — positional deps resolve to real IDs
    out.truncate(0); out.seek(0)
    C.cmd_plan_create(ctx, proposed)
    tasks = svc.list_tasks(db)
    assert len(tasks) == 2
    dev = next(t for t in tasks if t.agent_role == "development")
    test = next(t for t in tasks if t.agent_role == "testing")
    assert test.depends_on == [dev.task_id]

    # 3. Approve both at the before gate (in real life this would be
    # one-at-a-time; the test compresses that flow)
    C.cmd_gate_before_approve(ctx, dev.task_id)
    C.cmd_gate_before_approve(ctx, test.task_id)

    # 4. Pod processes the dev task
    cfg = load_config(paths.config_yaml)
    dev_fake = FakeAnthropic(
        '{"files_changed": ["main.py"], '
        '"rationale": "added greet() returning friendly string"}'
    )
    status = process_one_task(
        "pod_a", backend=ctx.backend,
        anthropic_caller=_caller_for(dev_fake), config=cfg,
    )
    assert status == "done"
    assert svc.get_task(db, dev.task_id).status == "after_gate"

    # 5. Approve dev at after gate
    C.cmd_gate_after_approve(ctx, dev.task_id)

    # 6. Update rolling summary (parent's responsibility)
    summary_md = paths.root / "summary_v1.md"
    summary_md.write_text(
        "## Goal\nAdd greet() function with tests\n\n"
        "## Completed milestones\n- main.py greet() implemented\n\n"
        "## Open threads\n- write tests\n"
    )
    C.cmd_summary_update(ctx, summary_md)

    # 7. Pod processes the testing task — it should see the updated
    # rolling summary (anti-context-rot mechanism)
    test_fake = FakeAnthropic(
        '{"tests_run": 3, "passed": 3, "failed": [], "runtime_seconds": 0.1}'
    )
    process_one_task(
        "pod_a", backend=ctx.backend,
        anthropic_caller=_caller_for(test_fake), config=cfg,
    )
    user_msg = test_fake.calls[0]["messages"][0]["content"]
    assert "main.py greet() implemented" in user_msg, (
        "testing pod should see the updated rolling summary"
    )

    # 8. Approve testing at after gate
    C.cmd_gate_after_approve(ctx, test.task_id)

    # 9. Final-state assertions
    states = {t.task_id: t.status for t in svc.list_tasks(db)}
    assert states[dev.task_id] == "done"
    assert states[test.task_id] == "done"

    # Per-agent change logs exist with the right entries
    dev_log = (paths.logs_dir / "development_agent.md").read_text()
    test_log = (paths.logs_dir / "testing_agent.md").read_text()
    assert dev.task_id in dev_log
    assert "greet()" in dev_log
    assert test.task_id in test_log
    assert "3 passed" in test_log

    # Budget ledger has two rows (one per pod call)
    rows = db.query_all("SELECT * FROM budget_ledger")
    assert len(rows) == 2

    # parent_actions has all the tools we used
    tools = {r["tool"] for r in db.query_all("SELECT tool FROM parent_actions")}
    expected = {
        "framework_subagent_invoke",
        "framework_plan_create",
        "framework_gate_before_approve",
        "framework_gate_after_approve",
        "framework_summary_update",
    }
    assert expected.issubset(tools), expected - tools
