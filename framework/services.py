"""Task and artifact service layer.

Holds the state-transition logic that the API endpoints call into.
Splitting it out keeps the FastAPI handlers thin and lets unit tests
exercise transitions without spinning up an HTTP client.

State machine (Section 8 of the methodology):

    created → before_gate → ready → claimed → running → after_gate → done
                  ↓                                          ↓
               rejected ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ rejected
                  ↑ (re-queue path: rejected_after → before_gate after edit)
"""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

from framework.db import Database, utcnow_iso
from framework.events import emit_event
from framework.models import (
    ArtifactCreate, ArtifactOut, FailureIn, SubmitResultIn,
    TaskCreate, TaskEdit, TaskOut,
)
from framework.scheduler import claim_next_task
from framework.state import StatePaths

_id_lock = threading.Lock()
_task_counter = 0
_artifact_counter = 0


def _new_task_id() -> str:
    global _task_counter
    with _id_lock:
        _task_counter += 1
        return f"t_{uuid.uuid4().hex[:10]}"


def _new_artifact_id(artifact_type: str) -> str:
    global _artifact_counter
    with _id_lock:
        _artifact_counter += 1
    suffix = artifact_type.lower()[:6] if artifact_type else "art"
    return f"a_{suffix}_{uuid.uuid4().hex[:8]}"


class TaskNotFound(Exception):
    pass


class IllegalTransition(Exception):
    pass


def _row_to_task(row) -> TaskOut:
    return TaskOut.from_row(row)


_log_lock = threading.Lock()


def _one_line_outcome(
    artifact_type: str, content: Any, fallback: str,
) -> str:
    """Synthesize a one-line outcome string for the per-agent log."""
    if not isinstance(content, dict):
        return fallback[:120]
    if artifact_type == "PatchSummary":
        rationale = content.get("rationale") or fallback
        return rationale.replace("\n", " ").strip()[:160]
    if artifact_type == "TestResult":
        return (
            f"{content.get('passed', '?')} passed, "
            f"{len(content.get('failed', []) or [])} failed of "
            f"{content.get('tests_run', '?')}"
        )
    if artifact_type == "ResearchBrief":
        return (content.get("summary") or fallback).replace("\n", " ").strip()[:160]
    if artifact_type == "FailureReport":
        return f"FAILED: {content.get('error_message', '')}".strip()[:160]
    if artifact_type == "ProgressLogEntry":
        return (content.get("summary") or fallback).replace("\n", " ").strip()[:160]
    return fallback.replace("\n", " ").strip()[:160]


def _files_touched(artifact_type: str, content: Any) -> list[str]:
    if isinstance(content, dict) and artifact_type == "PatchSummary":
        files = content.get("files_changed") or []
        return [str(f) for f in files if f]
    return []


def append_progress_log(
    logs_dir: Path,
    *,
    role: str,
    task_id: str,
    artifacts: list[dict[str, Any]],
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    fallback_outcome: str = "",
) -> None:
    """Append one ProgressLogEntry line to ``logs/<role>_agent.md``.

    Format (Section 9.3):
       ``timestamp | task_id | one-line outcome | tokens_in/out | cost_usd | files_touched | artifact_ids``
    """
    if not artifacts:
        return
    primary = artifacts[0]
    outcome = _one_line_outcome(
        primary.get("artifact_type", ""), primary.get("content"),
        fallback_outcome,
    )
    files = []
    for a in artifacts:
        files.extend(_files_touched(a.get("artifact_type", ""), a.get("content")))
    artifact_ids = [a["artifact_id"] for a in artifacts if a.get("artifact_id")]
    line = (
        f"{utcnow_iso()}"
        f" | {task_id}"
        f" | {outcome}"
        f" | {tokens_in}/{tokens_out}"
        f" | ${cost_usd:.4f}"
        f" | {','.join(files) if files else '-'}"
        f" | {','.join(artifact_ids) if artifact_ids else '-'}"
    )
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{role}_agent.md"
    header = f"# {role} agent — change log\n\n" if not log_file.exists() else ""
    with _log_lock:
        with log_file.open("a", encoding="utf-8") as f:
            if header:
                f.write(header)
            f.write(line + "\n")


def get_task(db: Database, task_id: str) -> TaskOut:
    row = db.query_one("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
    if row is None:
        raise TaskNotFound(task_id)
    return _row_to_task(row)


def list_tasks(
    db: Database,
    status: str | None = None,
    *,
    include_archived: bool = False,
) -> list[TaskOut]:
    """List tasks. Archived (post-session-reset) tasks are filtered out
    by default — pass ``include_archived=True`` to see them."""
    where = []
    params: list = []
    if status is not None:
        where.append("status = ?")
        params.append(status)
    if not include_archived:
        where.append("archived_at IS NULL")
    sql = "SELECT * FROM tasks"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY priority DESC, created_at ASC" if status else " ORDER BY created_at ASC"
    rows = db.query_all(sql, tuple(params))
    return [_row_to_task(r) for r in rows]


def session_reset(db: Database, events_jsonl: Path) -> dict[str, Any]:
    """Archive all done/rejected tasks so they drop out of the active
    queue display. SQLite rows are preserved for audit trail."""
    now = utcnow_iso()
    cur = db.execute(
        "UPDATE tasks SET archived_at = ? "
        "WHERE archived_at IS NULL AND status IN ('done', 'rejected')",
        (now, ),
    )
    archived = cur.rowcount
    emit_event(
        db, events_jsonl, "summary_updated",
        payload={"event": "session_reset", "archived_tasks": archived, "ts": now},
    )
    return {"archived_tasks": archived, "ts": now}


def requeue_task(
    db: Database, events_jsonl: Path, task_id: str,
) -> TaskOut:
    """Reset a stuck claimed/running task back to ready.

    Use when a pod was killed mid-task and the task needs to be picked
    up again. Pod assignment + timestamps are cleared. retry_count is
    bumped so the audit trail shows this happened.
    """
    task = get_task(db, task_id)
    if task.status not in ("claimed", "running"):
        raise IllegalTransition(
            f"task {task_id} is in {task.status!r}; "
            "requeue is only valid for 'claimed' or 'running'"
        )
    prev_pod = task.pod_id
    db.execute(
        "UPDATE tasks SET status = 'ready', "
        "pod_id = NULL, claimed_at = NULL, started_at = NULL, "
        "retry_count = retry_count + 1 "
        "WHERE task_id = ?",
        (task_id,),
    )
    if prev_pod:
        db.execute(
            "UPDATE pods SET status = 'idle', current_task_id = NULL, last_seen = ? "
            "WHERE pod_id = ? AND current_task_id = ?",
            (utcnow_iso(), prev_pod, task_id),
        )
    emit_event(
        db, events_jsonl, "plan_revised",
        task_id=task_id,
        payload={"event": "requeue", "from_pod": prev_pod},
    )
    return get_task(db, task_id)


def create_task(
    db: Database,
    events_jsonl: Path,
    spec: TaskCreate,
    *,
    initial_status: str = "before_gate",
) -> TaskOut:
    """Insert a new task. Defaults to ``before_gate`` so it surfaces to the user."""
    task_id = spec.task_id or _new_task_id()
    now = utcnow_iso()
    db.execute(
        """
        INSERT INTO tasks (
            task_id, parent_task_id, agent_role, goal_text,
            input_artifact_ids, output_artifact_types, recommended_model,
            priority, created_at, depends_on, working_dir, status
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            task_id,
            spec.parent_task_id,
            spec.agent_role,
            spec.goal_text,
            json.dumps(spec.input_artifact_ids),
            json.dumps(spec.output_artifact_types),
            spec.recommended_model,
            spec.priority,
            now,
            json.dumps(spec.depends_on),
            spec.working_dir,
            initial_status,
        ),
    )
    emit_event(
        db, events_jsonl, "task_created",
        task_id=task_id,
        payload={"agent_role": spec.agent_role, "priority": spec.priority},
    )
    if initial_status == "before_gate":
        emit_event(
            db, events_jsonl, "task_before_gate", task_id=task_id, payload={},
        )
    return get_task(db, task_id)


def edit_task(
    db: Database,
    events_jsonl: Path,
    task_id: str,
    edit: TaskEdit,
) -> TaskOut:
    task = get_task(db, task_id)
    if task.status != "before_gate":
        raise IllegalTransition(
            f"task {task_id} is in {task.status!r}; edits only allowed in 'before_gate'"
        )

    fields: dict[str, Any] = {}
    for name in (
        "goal_text", "agent_role", "recommended_model", "priority",
        "working_dir",
    ):
        v = getattr(edit, name)
        if v is not None:
            fields[name] = v
    for name in ("input_artifact_ids", "output_artifact_types", "depends_on"):
        v = getattr(edit, name)
        if v is not None:
            fields[name] = json.dumps(v)

    if not fields:
        return task

    set_clause = ", ".join(f"{k} = ?" for k in fields.keys())
    db.execute(
        f"UPDATE tasks SET {set_clause} WHERE task_id = ?",
        (*fields.values(), task_id),
    )
    emit_event(
        db, events_jsonl, "plan_revised",
        task_id=task_id,
        payload={"fields": list(fields.keys())},
    )
    return get_task(db, task_id)


def _load_run_meta(state_paths: StatePaths) -> dict[str, Any] | None:
    """Read run.yaml from a state-dir, or return None if missing/unreadable."""
    rp = state_paths.run_yaml
    if not rp.exists():
        return None
    try:
        import yaml
        return yaml.safe_load(rp.read_text(encoding="utf-8")) or {}
    except Exception:
        import logging
        logging.getLogger(__name__).exception(
            "could not read run.yaml at %s", rp,
        )
        return None


def _maybe_create_worktree_on_approve(
    db: Database, events_jsonl: Path, state_paths: StatePaths, task: TaskOut,
) -> None:
    """v2: development tasks get a per-task git worktree on before-gate
    approve so two pods can edit the same logical repo via separate
    checkouts. Testing tasks that depend on a dev task with a worktree
    are pointed at the same worktree (read-only by role contract).

    No-op when:
    - the task isn't development or testing
    - run.yaml is missing or target isn't a git repo
    - imports fail (worktree module deps)
    """
    if task.agent_role not in ("development", "testing"):
        return
    meta = _load_run_meta(state_paths)
    if not meta or not meta.get("target_is_git"):
        return
    target_repo = meta.get("target_repo")
    base_branch = meta.get("branch_name")
    if not target_repo or not base_branch:
        return

    from framework.worktree import WorktreeError, create_worktree

    # Both dev and testing get a fresh worktree from the base branch.
    # By the time a testing task is approved, its dev dependencies have
    # already been approved at after-gate (merged into base), so the
    # base branch reflects the latest approved state. Testing reads
    # but doesn't write — its `allowed_tools` excludes filesystem_write
    # — so the contract is enforced by the pod, not by branch isolation.
    try:
        wt = create_worktree(
            target_repo, base_branch, task.task_id,
            state_paths.worktrees_dir,
        )
    except WorktreeError as e:
        import logging
        logging.getLogger(__name__).warning(
            "worktree creation failed for %s: %s", task.task_id, e,
        )
        return
    db.execute(
        "UPDATE tasks SET working_dir = ?, worktree_path = ? "
        "WHERE task_id = ?",
        (str(wt), str(wt), task.task_id),
    )
    emit_event(
        db, events_jsonl, "worktree_created",
        task_id=task.task_id,
        payload={"worktree_path": str(wt),
                 "branch": f"{base_branch}-{task.task_id}",
                 "role": task.agent_role},
    )


def approve_before(
    db: Database, events_jsonl: Path, task_id: str,
    *, state_paths: StatePaths | None = None,
) -> TaskOut:
    task = get_task(db, task_id)
    if task.status != "before_gate":
        raise IllegalTransition(
            f"task {task_id} is in {task.status!r}; expected 'before_gate'"
        )
    db.execute("UPDATE tasks SET status = 'ready' WHERE task_id = ?", (task_id,))
    emit_event(db, events_jsonl, "task_approved_before", task_id=task_id)
    if state_paths is not None:
        _maybe_create_worktree_on_approve(
            db, events_jsonl, state_paths, get_task(db, task_id),
        )
    return get_task(db, task_id)


def reject_before(
    db: Database, events_jsonl: Path, task_id: str, reason: str
) -> TaskOut:
    task = get_task(db, task_id)
    if task.status != "before_gate":
        raise IllegalTransition(
            f"task {task_id} is in {task.status!r}; expected 'before_gate'"
        )
    db.execute(
        "UPDATE tasks SET status = 'rejected', rejection_reason = ? "
        "WHERE task_id = ?",
        (reason, task_id),
    )
    emit_event(
        db, events_jsonl, "task_rejected_before",
        task_id=task_id, payload={"reason": reason},
    )
    return get_task(db, task_id)


def claim(
    db: Database, events_jsonl: Path, pod_id: str,
    *, daily_cap_usd: float | None = None,
) -> TaskOut | None:
    raw = claim_next_task(
        db, pod_id, events_jsonl, daily_cap_usd=daily_cap_usd,
    )
    if raw is None:
        return None
    return get_task(db, raw["task_id"])


def mark_running(
    db: Database, events_jsonl: Path, task_id: str
) -> TaskOut:
    task = get_task(db, task_id)
    if task.status != "claimed":
        raise IllegalTransition(
            f"task {task_id} is in {task.status!r}; expected 'claimed'"
        )
    db.execute(
        "UPDATE tasks SET status = 'running', started_at = ? WHERE task_id = ?",
        (utcnow_iso(), task_id),
    )
    return get_task(db, task_id)


def submit_result(
    db: Database,
    events_jsonl: Path,
    budget_jsonl: Path,
    task_id: str,
    result: SubmitResultIn,
    *,
    logs_dir: Path | None = None,
) -> tuple[TaskOut, list[ArtifactOut]]:
    task = get_task(db, task_id)
    if task.status not in ("running", "claimed"):
        raise IllegalTransition(
            f"task {task_id} is in {task.status!r}; expected 'running' or 'claimed'"
        )

    artifacts_out: list[ArtifactOut] = []
    now = utcnow_iso()
    for art in result.artifacts:
        artifact_id = art.artifact_id or _new_artifact_id(art.artifact_type)
        db.execute(
            """
            INSERT INTO artifacts (
                artifact_id, artifact_type, produced_by_task, produced_by_agent,
                produced_at, tokens_in, tokens_out, cost_usd, duration_seconds,
                model, content
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                artifact_id,
                art.artifact_type,
                art.produced_by_task,
                art.produced_by_agent,
                now,
                art.tokens_in,
                art.tokens_out,
                art.cost_usd,
                art.duration_seconds,
                art.model,
                json.dumps(art.content),
            ),
        )
        emit_event(
            db, events_jsonl, "artifact_submitted",
            task_id=task_id, payload={"artifact_id": artifact_id, "type": art.artifact_type},
        )
        row = db.query_one(
            "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
        )
        artifacts_out.append(ArtifactOut.from_row(row))

    db.execute(
        "INSERT INTO budget_ledger ("
        "ts, pod_id, task_id, agent_role, model, "
        "input_tokens, output_tokens, cost_usd, duration_seconds) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (
            now,
            task.pod_id or "unknown",
            task_id,
            task.agent_role,
            result.model,
            result.input_tokens,
            result.output_tokens,
            result.cost_usd,
            result.duration_seconds,
        ),
    )
    budget_line = json.dumps({
        "ts": now,
        "pod_id": task.pod_id,
        "task_id": task_id,
        "agent_role": task.agent_role,
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cost_usd": result.cost_usd,
        "duration_seconds": result.duration_seconds,
    }, sort_keys=True)
    budget_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with budget_jsonl.open("a", encoding="utf-8") as f:
        f.write(budget_line + "\n")
    emit_event(
        db, events_jsonl, "budget_updated",
        task_id=task_id,
        payload={
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": result.cost_usd,
        },
    )

    db.execute(
        "UPDATE tasks SET status = 'after_gate', completed_at = ? WHERE task_id = ?",
        (now, task_id),
    )
    emit_event(db, events_jsonl, "task_completed", task_id=task_id)
    emit_event(db, events_jsonl, "task_after_gate", task_id=task_id)

    if task.pod_id:
        db.execute(
            "UPDATE pods SET status = 'idle', current_task_id = NULL, last_seen = ? "
            "WHERE pod_id = ?",
            (now, task.pod_id),
        )

    # Per-agent change log (Phase 5). Best-effort: a logging failure
    # must not invalidate the just-saved artifact + ledger row.
    if logs_dir is not None:
        try:
            append_progress_log(
                logs_dir,
                role=task.agent_role,
                task_id=task_id,
                artifacts=[a.model_dump() for a in artifacts_out],
                tokens_in=result.input_tokens,
                tokens_out=result.output_tokens,
                cost_usd=result.cost_usd,
                fallback_outcome=task.goal_text,
            )
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "append_progress_log failed for task %s", task_id,
            )

    return get_task(db, task_id), artifacts_out


def _maybe_cleanup_worktree(
    db: Database, events_jsonl: Path, state_paths: StatePaths,
    task: TaskOut, *, capture_diff: bool,
) -> str | None:
    """If the task owns a worktree (i.e. ``worktree_path`` and is a dev
    task), tear it down. Optionally extract a diff first so the
    PatchSummary can record what changed. Best-effort — the gate
    transition has already happened by the time this runs.

    Testing-role worktrees are removed but not merged (read-only by
    role contract; nothing to push back to base).
    """
    if task.agent_role not in ("development", "testing"):
        return None
    if not task.worktree_path:
        return None
    meta = _load_run_meta(state_paths)
    if not meta:
        return None
    target_repo = meta.get("target_repo")
    base_branch = meta.get("branch_name")
    if not target_repo or not base_branch:
        return None

    diff_text: str | None = None
    try:
        from framework.worktree import (
            auto_commit_all, extract_diff, merge_into_base, remove_worktree,
        )
        if capture_diff and task.agent_role == "development":
            # Auto-commit any pod edits the model didn't commit itself,
            # so the per-task branch reflects the work and the merge
            # has something to fast-forward.
            auto_commit_all(
                task.worktree_path,
                f"[{task.task_id}] auto-commit on after-gate approve",
            )
            diff_text = extract_diff(task.worktree_path, base_branch) or None
            # Merge the per-task branch into base so downstream tasks
            # (whose worktrees fork from base at before-gate approve)
            # see the just-approved changes.
            task_branch = f"{base_branch}-{task.task_id}"
            try:
                merge_into_base(target_repo, base_branch, task_branch)
                emit_event(
                    db, events_jsonl, "plan_revised",
                    task_id=task.task_id,
                    payload={"event": "merged_into_base",
                             "branch": task_branch, "base": base_branch},
                )
            except Exception as merge_err:
                import logging
                logging.getLogger(__name__).warning(
                    "merge into %s failed for %s: %s — branch left dangling",
                    base_branch, task.task_id, merge_err,
                )
        remove_worktree(target_repo, task.worktree_path)
    except Exception:
        import logging
        logging.getLogger(__name__).exception(
            "worktree cleanup failed for %s", task.task_id,
        )
        return diff_text

    db.execute(
        "UPDATE tasks SET worktree_path = NULL WHERE task_id = ?",
        (task.task_id,),
    )
    emit_event(
        db, events_jsonl, "worktree_removed",
        task_id=task.task_id,
        payload={"had_diff": bool(diff_text)},
    )
    return diff_text


def _attach_diff_to_patch_summary(
    db: Database, task_id: str, diff_text: str,
) -> None:
    """Append the worktree diff to the task's most recent PatchSummary
    artifact's content. No-op if no PatchSummary exists.
    """
    row = db.query_one(
        "SELECT artifact_id, content FROM artifacts "
        "WHERE produced_by_task = ? AND artifact_type = 'PatchSummary' "
        "ORDER BY produced_at DESC LIMIT 1",
        (task_id,),
    )
    if row is None:
        return
    try:
        content = json.loads(row["content"])
    except (TypeError, json.JSONDecodeError):
        return
    if not isinstance(content, dict):
        return
    content["diff"] = diff_text
    db.execute(
        "UPDATE artifacts SET content = ? WHERE artifact_id = ?",
        (json.dumps(content), row["artifact_id"]),
    )


def approve_after(
    db: Database, events_jsonl: Path, task_id: str,
    *, state_paths: StatePaths | None = None,
) -> TaskOut:
    task = get_task(db, task_id)
    if task.status != "after_gate":
        raise IllegalTransition(
            f"task {task_id} is in {task.status!r}; expected 'after_gate'"
        )
    db.execute("UPDATE tasks SET status = 'done' WHERE task_id = ?", (task_id,))
    emit_event(db, events_jsonl, "task_approved_after", task_id=task_id)
    if state_paths is not None:
        diff = _maybe_cleanup_worktree(
            db, events_jsonl, state_paths, task, capture_diff=True,
        )
        if diff:
            _attach_diff_to_patch_summary(db, task_id, diff)
    return get_task(db, task_id)


def reject_after(
    db: Database, events_jsonl: Path, task_id: str, reason: str,
    *, state_paths: StatePaths | None = None,
) -> TaskOut:
    """Reject artifact at the after gate; task returns to ``before_gate``
    so the user can edit the spec and retry. ``retry_count`` is incremented.

    The worktree is torn down — the next approve gets a fresh one. Any
    diff in the rejected worktree is lost (the user already saw the
    artifact and decided it was wrong; preserving the diff would just
    leak stale state into the retry).
    """
    task = get_task(db, task_id)
    if task.status != "after_gate":
        raise IllegalTransition(
            f"task {task_id} is in {task.status!r}; expected 'after_gate'"
        )
    db.execute(
        "UPDATE tasks SET status = 'before_gate', "
        "rejection_reason = ?, retry_count = retry_count + 1, "
        "pod_id = NULL, claimed_at = NULL, started_at = NULL, completed_at = NULL "
        "WHERE task_id = ?",
        (reason, task_id),
    )
    emit_event(
        db, events_jsonl, "task_rejected_after",
        task_id=task_id, payload={"reason": reason},
    )
    emit_event(db, events_jsonl, "task_before_gate", task_id=task_id)
    if state_paths is not None:
        _maybe_cleanup_worktree(
            db, events_jsonl, state_paths, task, capture_diff=False,
        )
    return get_task(db, task_id)


def report_failure(
    db: Database,
    events_jsonl: Path,
    task_id: str,
    failure: FailureIn,
    *,
    logs_dir: Path | None = None,
) -> TaskOut:
    """Pod-side failure path: write a FailureReport artifact and surface it
    at the after gate, exactly like a normal artifact (per Section 16).
    """
    task = get_task(db, task_id)
    if task.status not in ("running", "claimed"):
        raise IllegalTransition(
            f"task {task_id} is in {task.status!r}; expected 'running' or 'claimed'"
        )
    artifact_id = _new_artifact_id("Failur")
    now = utcnow_iso()
    content = {
        "failure_mode": failure.failure_mode,
        "error_message": failure.error_message,
        "retry_count": failure.retry_count,
        "recommended_action": "retry",
    }
    db.execute(
        """
        INSERT INTO artifacts (
            artifact_id, artifact_type, produced_by_task, produced_by_agent,
            produced_at, content
        ) VALUES (?,?,?,?,?,?)
        """,
        (artifact_id, "FailureReport", task_id, task.agent_role, now,
         json.dumps(content)),
    )
    emit_event(
        db, events_jsonl, "artifact_submitted",
        task_id=task_id, payload={"artifact_id": artifact_id, "type": "FailureReport"},
    )
    db.execute(
        "UPDATE tasks SET status = 'after_gate', completed_at = ? WHERE task_id = ?",
        (now, task_id),
    )
    emit_event(db, events_jsonl, "task_after_gate", task_id=task_id)
    if task.pod_id:
        db.execute(
            "UPDATE pods SET status = 'idle', current_task_id = NULL, last_seen = ? "
            "WHERE pod_id = ?",
            (now, task.pod_id),
        )

    if logs_dir is not None:
        try:
            append_progress_log(
                logs_dir,
                role=task.agent_role,
                task_id=task_id,
                artifacts=[{
                    "artifact_id": artifact_id,
                    "artifact_type": "FailureReport",
                    "content": content,
                }],
                tokens_in=0, tokens_out=0, cost_usd=0.0,
                fallback_outcome=task.goal_text,
            )
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "append_progress_log failed for failure on task %s", task_id,
            )

    return get_task(db, task_id)


# --- Artifact helpers --------------------------------------------------

def get_artifact(db: Database, artifact_id: str) -> ArtifactOut:
    row = db.query_one(
        "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
    )
    if row is None:
        raise TaskNotFound(artifact_id)
    return ArtifactOut.from_row(row)


def list_artifacts(
    db: Database,
    *,
    artifact_type: str | None = None,
    task_id: str | None = None,
) -> list[ArtifactOut]:
    sql = "SELECT * FROM artifacts WHERE 1=1"
    params: list[Any] = []
    if artifact_type is not None:
        sql += " AND artifact_type = ?"
        params.append(artifact_type)
    if task_id is not None:
        sql += " AND produced_by_task = ?"
        params.append(task_id)
    sql += " ORDER BY produced_at ASC"
    rows = db.query_all(sql, tuple(params))
    return [ArtifactOut.from_row(r) for r in rows]


# --- Pod helpers --------------------------------------------------------

def register_pod(db: Database, events_jsonl: Path, pod_id: str) -> dict[str, Any]:
    now = utcnow_iso()
    db.execute(
        "INSERT INTO pods (pod_id, status, last_seen, registered_at) "
        "VALUES (?, 'idle', ?, ?) "
        "ON CONFLICT(pod_id) DO UPDATE SET status='idle', last_seen=excluded.last_seen",
        (pod_id, now, now),
    )
    emit_event(db, events_jsonl, "pod_registered",
               payload={"pod_id": pod_id})
    row = db.query_one("SELECT * FROM pods WHERE pod_id = ?", (pod_id,))
    return dict(row)


def list_pods(db: Database) -> list[dict[str, Any]]:
    return [dict(r) for r in db.query_all("SELECT * FROM pods ORDER BY pod_id")]
