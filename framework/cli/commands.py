"""Framework tool command implementations.

Each function corresponds to one ``framework <subcommand>`` from
Section 5.2. They are kept as plain functions (not argparse-coupled) so
tests can call them directly with a ``CliContext``.

Convention: every function returns an int exit code (0 on success) and
records an entry in ``parent_actions.jsonl`` via ``ctx.log_action``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from framework.bootstrap import bootstrap_run
from framework.cli._context import CliContext


# ---------------- helpers --------------------------------------------

def _print_yaml(ctx: CliContext, obj: Any) -> None:
    ctx.stdout.write(yaml.safe_dump(obj, sort_keys=False, default_flow_style=False))


def _print_json(ctx: CliContext, obj: Any) -> None:
    ctx.stdout.write(json.dumps(obj, indent=2, sort_keys=False))
    ctx.stdout.write("\n")


# ---------------- inspection ------------------------------------------

def cmd_state(ctx: CliContext, *, recent_events: int = 10) -> int:
    state = ctx.backend.get_state(recent_events=recent_events)
    ctx.log_action("framework_state", {"recent_events": recent_events})
    _print_yaml(ctx, state)
    return 0


def cmd_db_query(ctx: CliContext, sql: str) -> int:
    result = ctx.backend.db_query(sql)
    ctx.log_action("framework_db_query", {"sql": sql,
                                          "row_count": result["row_count"]})
    _print_yaml(ctx, result)
    return 0


def cmd_artifact_get(ctx: CliContext, artifact_id: str) -> int:
    art = ctx.backend.get_artifact(artifact_id)
    ctx.log_action("framework_artifact_get", {"artifact_id": artifact_id})
    _print_yaml(ctx, art)
    return 0


def cmd_artifact_list(
    ctx: CliContext, *, type: str | None = None, task_id: str | None = None,
) -> int:
    arts = ctx.backend.list_artifacts(type=type, task_id=task_id)
    summary = [
        {
            "artifact_id": a["artifact_id"],
            "artifact_type": a["artifact_type"],
            "produced_by_task": a["produced_by_task"],
            "produced_by_agent": a["produced_by_agent"],
            "produced_at": a["produced_at"],
            "model": a["model"],
            "tokens_in": a["tokens_in"],
            "tokens_out": a["tokens_out"],
            "cost_usd": a["cost_usd"],
        }
        for a in arts
    ]
    ctx.log_action("framework_artifact_list",
                   {"type": type, "task_id": task_id, "count": len(arts)})
    _print_yaml(ctx, summary)
    return 0


def cmd_plan_show(
    ctx: CliContext, *, status: str | None = None, include_archived: bool = False,
) -> int:
    tasks = ctx.backend.list_tasks(status=status, include_archived=include_archived)
    summary = [
        {
            "task_id": t["task_id"],
            "status": t["status"],
            "agent_role": t["agent_role"],
            "priority": t["priority"],
            "recommended_model": t["recommended_model"],
            "goal_text": (t["goal_text"][:120] + "…")
                        if len(t["goal_text"]) > 120 else t["goal_text"],
            "depends_on": t["depends_on"],
        }
        for t in tasks
    ]
    ctx.log_action("framework_plan_show",
                   {"status": status, "count": len(tasks),
                    "include_archived": include_archived})
    _print_yaml(ctx, summary)
    return 0


# ---------------- session / resilience -------------------------------

def cmd_session_reset(ctx: CliContext) -> int:
    """Archive all done/rejected tasks so they drop out of the active queue."""
    res = ctx.backend.session_reset()
    ctx.log_action("framework_session_reset",
                   {"archived_tasks": res["archived_tasks"]})
    ctx.stdout.write(
        f"archived {res['archived_tasks']} done/rejected tasks "
        f"(audit trail preserved in SQLite + events.jsonl)\n"
    )
    return 0


def cmd_task_requeue(ctx: CliContext, task_id: str) -> int:
    """Reset a stuck claimed/running task back to ready.

    Use after a pod was killed mid-task. ``retry_count`` is bumped so
    the audit trail records the manual intervention.
    """
    t = ctx.backend.requeue_task(task_id)
    ctx.log_action("framework_task_requeue", {"task_id": task_id})
    ctx.stdout.write(
        f"task {task_id} → {t['status']} "
        f"(retry_count={t['retry_count']}, pod cleared)\n"
    )
    return 0


# ---------------- mutation ------------------------------------------

def _load_plan_yaml(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "tasks" in raw:
        tasks = raw["tasks"]
    elif isinstance(raw, list):
        tasks = raw
    else:
        raise ValueError(
            f"plan YAML must be a list or have a top-level 'tasks:' key (got {type(raw).__name__})"
        )
    if not isinstance(tasks, list):
        raise ValueError("'tasks' must be a list")
    return tasks


def cmd_plan_create(ctx: CliContext, yaml_file: str | Path) -> int:
    """Create tasks from a YAML plan.

    ``depends_on`` entries can be either a real ``task_id`` string or an
    integer position (the methodology agent emits the latter, since it
    doesn't know task IDs ahead of time). Positions are resolved to real
    task IDs as tasks are created in order; forward / self / out-of-range
    references are dropped with a warning.
    """
    tasks = _load_plan_yaml(yaml_file)
    created = []
    real_ids: list[str] = []
    for i, spec in enumerate(tasks):
        spec = dict(spec)  # don't mutate caller's dict
        deps = spec.get("depends_on") or []
        resolved: list[str] = []
        for d in deps:
            if isinstance(d, int):
                if 0 <= d < i:
                    resolved.append(real_ids[d])
                else:
                    ctx.stderr.write(
                        f"warning: task[{i}] depends_on={d} out of range; dropping\n"
                    )
            elif isinstance(d, str):
                resolved.append(d)
        spec["depends_on"] = resolved
        t = ctx.backend.create_task(spec, initial_status="before_gate")
        real_ids.append(t["task_id"])
        created.append({
            "position": i,
            "task_id": t["task_id"],
            "agent_role": t["agent_role"],
            "depends_on": resolved,
        })
    ctx.log_action("framework_plan_create",
                   {"yaml_file": str(yaml_file), "created": len(created)})
    _print_yaml(ctx, {"created": created})
    return 0


def cmd_plan_edit(ctx: CliContext, task_id: str, field: str, value: str) -> int:
    """Edit one field of a TaskSpec at the before gate.

    The CLI accepts the value as a string and parses it appropriately
    based on the field. List/JSON fields accept either YAML or JSON.
    """
    parsed: Any = value
    list_fields = {"input_artifact_ids", "output_artifact_types", "depends_on"}
    int_fields = {"priority"}
    if field in list_fields:
        parsed = yaml.safe_load(value) if value else []
        if not isinstance(parsed, list):
            raise ValueError(f"{field} must be a list; got {type(parsed).__name__}")
    elif field in int_fields:
        parsed = int(value)

    body = {field: parsed}
    updated = ctx.backend.edit_task(task_id, body)
    ctx.log_action("framework_plan_edit",
                   {"task_id": task_id, "field": field})
    _print_yaml(ctx, {
        "task_id": updated["task_id"],
        "status": updated["status"],
        field: updated[field],
    })
    return 0


def cmd_gate_before_approve(
    ctx: CliContext, task_id: str | list[str],
) -> int:
    """Approve one or more tasks at the before gate.

    Accepting a list lets the caller flip several tasks to ``ready`` in
    a single ``python -m framework`` invocation, which collapses the
    ~1–2s Python boot time per task into one. Concretely: when two
    independent dev tasks share no dependencies, batching their
    approval lets two pods polling at 2s actually claim them in the
    same window — wall-clock parallelism that sequential per-task
    invocations miss because of the boot-overhead gap.
    """
    task_ids = [task_id] if isinstance(task_id, str) else list(task_id)
    if not task_ids:
        ctx.stderr.write("no task_ids provided\n")
        return 2
    for tid in task_ids:
        t = ctx.backend.approve_before(tid)
        ctx.log_action("framework_gate_before_approve", {"task_id": tid})
        ctx.stdout.write(f"task {tid} → {t['status']}\n")
    return 0


def cmd_gate_before_reject(ctx: CliContext, task_id: str, reason: str) -> int:
    t = ctx.backend.reject_before(task_id, reason)
    ctx.log_action("framework_gate_before_reject",
                   {"task_id": task_id, "reason": reason})
    ctx.stdout.write(f"task {task_id} → {t['status']} ({reason})\n")
    return 0


def cmd_gate_after_approve(ctx: CliContext, task_id: str) -> int:
    t = ctx.backend.approve_after(task_id)
    ctx.log_action("framework_gate_after_approve", {"task_id": task_id})
    ctx.stdout.write(f"task {task_id} → {t['status']}\n")
    ctx.stdout.write(
        "reminder: read the just-approved artifact + the prior "
        "rolling_summary.md, produce a new summary, and run "
        "`framework summary update <file>`.\n"
    )
    return 0


def cmd_gate_after_reject(ctx: CliContext, task_id: str, reason: str) -> int:
    t = ctx.backend.reject_after(task_id, reason)
    ctx.log_action("framework_gate_after_reject",
                   {"task_id": task_id, "reason": reason})
    ctx.stdout.write(
        f"task {task_id} → {t['status']} (retry_count={t['retry_count']})\n"
        f"reason: {reason}\n"
    )
    return 0


def cmd_summary_update(ctx: CliContext, summary_file: str | Path) -> int:
    p = Path(summary_file)
    content = p.read_text(encoding="utf-8")
    word_count = len(content.split())
    if word_count > 2200:  # rough proxy for the 2000-token cap
        ctx.stderr.write(
            f"warning: summary is ~{word_count} words; methodology caps at "
            "~2000 tokens. Drop oldest 'completed milestones' first.\n"
        )
    res = ctx.backend.update_summary(content)
    ctx.log_action("framework_summary_update",
                   {"source": str(p), "bytes": res["bytes"]})
    ctx.stdout.write(f"rolling_summary.md updated ({res['bytes']} bytes)\n")
    return 0


def cmd_run_start(
    ctx: CliContext, *, goal: str, target_repo: str, overwrite: bool = False,
) -> int:
    """Bootstrap a new run.

    This one is special: the framework-state directory may not exist yet,
    so the BackendClient cannot be used until *after* bootstrap. The
    parent_actions log goes to the freshly-created state dir via a fresh
    backend call performed at the end (when the backend, if running,
    can see the new DB).
    """
    info = bootstrap_run(
        ctx.paths.root, goal=goal, target_repo=target_repo, overwrite=overwrite,
    )
    # Logging the action requires the backend to be pointed at the new
    # state dir. If a backend was passed in (test mode) and is bound to
    # the same DB, this works. Otherwise we skip the HTTP log; the
    # bootstrap itself is a one-shot setup step the user can re-run.
    try:
        ctx.log_action("framework_run_start", {
            "goal": goal, "target_repo": str(target_repo), "run_id": info["run_id"],
        })
    except Exception:
        pass
    _print_yaml(ctx, info)
    return 0


# framework_subagent_invoke lives in framework/cli/subagent.py — re-export
# for parser dispatch.
from framework.cli.subagent import cmd_subagent_invoke  # noqa: E402, F401
