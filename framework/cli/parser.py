"""argparse wiring for the framework CLI.

Subcommands map 1:1 to ``cli.commands.cmd_*`` functions. Each parser
``set_defaults(func=...)`` so dispatch is uniform.
"""
from __future__ import annotations

import argparse

from framework.cli import commands as C


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="framework")
    p.add_argument(
        "--state-dir", default=None,
        help="framework-state/ directory (defaults to $FRAMEWORK_STATE_DIR or ./framework-state)",
    )
    p.add_argument(
        "--backend-url", default=None,
        help="backend base URL (defaults to $FRAMEWORK_BACKEND_URL or http://127.0.0.1:8765)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- existing admin commands ------------------------------------
    p_be = sub.add_parser("backend", help="start FastAPI backend")
    p_be.add_argument("--host", default="127.0.0.1")
    p_be.add_argument("--port", type=int, default=8765)
    p_be.set_defaults(_kind="admin", _name="backend")

    p_init = sub.add_parser("initdb", help="create state dir + db")
    p_init.set_defaults(_kind="admin", _name="initdb")

    p_pod = sub.add_parser("start-pod", help="run a pod worker")
    p_pod.add_argument("pod_id")
    p_pod.add_argument(
        "--api-key-env", default=None,
        help="env var name holding the pod's Anthropic API key. "
             "Defaults to ANTHROPIC_API_KEY_POD_<ID> (e.g. pod_b → "
             "ANTHROPIC_API_KEY_POD_B), with fallback to ANTHROPIC_API_KEY.",
    )
    p_pod.set_defaults(_kind="admin", _name="start-pod")

    # ---- inspection -------------------------------------------------
    p_state = sub.add_parser("state", help="snapshot of pods/queues/events/gates")
    p_state.add_argument("--recent-events", type=int, default=10)
    p_state.set_defaults(_kind="cli",
                         func=lambda ctx, a: C.cmd_state(ctx, recent_events=a.recent_events))

    p_db = sub.add_parser("db", help="db helpers")
    db_sub = p_db.add_subparsers(dest="db_cmd", required=True)
    p_db_q = db_sub.add_parser("query", help="read-only SQL")
    p_db_q.add_argument("sql")
    p_db_q.set_defaults(_kind="cli",
                        func=lambda ctx, a: C.cmd_db_query(ctx, a.sql))

    p_art = sub.add_parser("artifact", help="artifact helpers")
    art_sub = p_art.add_subparsers(dest="artifact_cmd", required=True)
    p_art_get = art_sub.add_parser("get", help="full artifact body by id")
    p_art_get.add_argument("artifact_id")
    p_art_get.set_defaults(_kind="cli",
                           func=lambda ctx, a: C.cmd_artifact_get(ctx, a.artifact_id))
    p_art_list = art_sub.add_parser("list", help="list artifacts")
    p_art_list.add_argument("--type", default=None)
    p_art_list.add_argument("--task", default=None, dest="task_id")
    p_art_list.set_defaults(
        _kind="cli",
        func=lambda ctx, a: C.cmd_artifact_list(ctx, type=a.type, task_id=a.task_id),
    )

    p_plan = sub.add_parser("plan", help="task plan helpers")
    plan_sub = p_plan.add_subparsers(dest="plan_cmd", required=True)
    p_plan_show = plan_sub.add_parser("show")
    p_plan_show.add_argument("--status", default=None)
    p_plan_show.add_argument("--include-archived", action="store_true",
                             help="show tasks archived by `framework session reset`")
    p_plan_show.set_defaults(
        _kind="cli",
        func=lambda ctx, a: C.cmd_plan_show(
            ctx, status=a.status, include_archived=a.include_archived,
        ),
    )
    p_plan_create = plan_sub.add_parser("create")
    p_plan_create.add_argument("yaml_file")
    p_plan_create.set_defaults(_kind="cli",
                               func=lambda ctx, a: C.cmd_plan_create(ctx, a.yaml_file))
    p_plan_edit = plan_sub.add_parser("edit")
    p_plan_edit.add_argument("task_id")
    p_plan_edit.add_argument("field")
    p_plan_edit.add_argument("value")
    p_plan_edit.set_defaults(
        _kind="cli",
        func=lambda ctx, a: C.cmd_plan_edit(ctx, a.task_id, a.field, a.value),
    )

    # ---- gates ------------------------------------------------------
    p_gate = sub.add_parser("gate", help="gate transitions")
    gate_sub = p_gate.add_subparsers(dest="gate_when", required=True)

    p_before = gate_sub.add_parser("before")
    before_sub = p_before.add_subparsers(dest="before_action", required=True)
    p_before_app = before_sub.add_parser(
        "approve",
        help="approve one or more tasks at the before gate. Pass multiple "
             "task IDs to flip them all to 'ready' in one invocation, so "
             "two pods can claim independent tasks in the same poll window.",
    )
    p_before_app.add_argument("task_id", nargs="+")
    p_before_app.set_defaults(_kind="cli",
                              func=lambda ctx, a: C.cmd_gate_before_approve(ctx, a.task_id))
    p_before_rej = before_sub.add_parser("reject")
    p_before_rej.add_argument("task_id")
    p_before_rej.add_argument("--reason", required=True)
    p_before_rej.set_defaults(
        _kind="cli",
        func=lambda ctx, a: C.cmd_gate_before_reject(ctx, a.task_id, a.reason),
    )

    p_after = gate_sub.add_parser("after")
    after_sub = p_after.add_subparsers(dest="after_action", required=True)
    p_after_app = after_sub.add_parser("approve")
    p_after_app.add_argument("task_id")
    p_after_app.set_defaults(_kind="cli",
                             func=lambda ctx, a: C.cmd_gate_after_approve(ctx, a.task_id))
    p_after_rej = after_sub.add_parser("reject")
    p_after_rej.add_argument("task_id")
    p_after_rej.add_argument("--reason", required=True)
    p_after_rej.set_defaults(
        _kind="cli",
        func=lambda ctx, a: C.cmd_gate_after_reject(ctx, a.task_id, a.reason),
    )

    # ---- summary + run ----------------------------------------------
    p_sum = sub.add_parser("summary", help="rolling summary helpers")
    sum_sub = p_sum.add_subparsers(dest="summary_cmd", required=True)
    p_sum_up = sum_sub.add_parser("update")
    p_sum_up.add_argument("summary_file")
    p_sum_up.set_defaults(_kind="cli",
                          func=lambda ctx, a: C.cmd_summary_update(ctx, a.summary_file))

    p_run = sub.add_parser("run", help="run lifecycle")
    run_sub = p_run.add_subparsers(dest="run_cmd", required=True)
    p_run_start = run_sub.add_parser("start", help="bootstrap framework-state/")
    p_run_start.add_argument("--goal", required=True)
    p_run_start.add_argument("--target-repo", required=True)
    p_run_start.add_argument("--overwrite", action="store_true")
    p_run_start.set_defaults(
        _kind="cli",
        func=lambda ctx, a: C.cmd_run_start(
            ctx, goal=a.goal, target_repo=a.target_repo, overwrite=a.overwrite,
        ),
    )

    # ---- session + task helpers (Phase 6) ---------------------------
    p_session = sub.add_parser("session", help="session lifecycle helpers")
    session_sub = p_session.add_subparsers(dest="session_cmd", required=True)
    p_session_reset = session_sub.add_parser(
        "reset",
        help="archive done/rejected tasks for cleaner active-queue display",
    )
    p_session_reset.set_defaults(_kind="cli",
                                 func=lambda ctx, a: C.cmd_session_reset(ctx))

    p_task = sub.add_parser("task", help="single-task helpers")
    task_sub = p_task.add_subparsers(dest="task_cmd", required=True)
    p_task_requeue = task_sub.add_parser(
        "requeue",
        help="reset a stuck claimed/running task back to ready (after pod kill)",
    )
    p_task_requeue.add_argument("task_id")
    p_task_requeue.set_defaults(_kind="cli",
                                func=lambda ctx, a: C.cmd_task_requeue(ctx, a.task_id))

    # ---- subagent ---------------------------------------------------
    p_sub_inv = sub.add_parser("subagent", help="synchronously invoke a subagent role")
    sub_inv = p_sub_inv.add_subparsers(dest="subagent_cmd", required=True)
    p_inv = sub_inv.add_parser("invoke")
    p_inv.add_argument("role", help="subagent role (Phase 4: methodology only)")
    p_inv.add_argument("task_yaml", help="path to a YAML file with at least a 'goal' field")
    p_inv.add_argument(
        "--api-key-env", default="ANTHROPIC_API_KEY",
        help="env var holding the API key for this subagent call",
    )
    p_inv.set_defaults(
        _kind="cli",
        func=lambda ctx, a: C.cmd_subagent_invoke(
            ctx, a.role, a.task_yaml, api_key_env=a.api_key_env,
        ),
    )

    return p
