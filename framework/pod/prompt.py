"""Prompt construction.

Phase 5: load ``framework-state/agents/<role>.md`` as the system prompt
and inject the per-task variables (goal, working_dir, rolling_summary,
input_artifacts) into the user message. The agent .md is fetched fresh
per task per Section 10 — no caching.

The Phase 2 hardcoded path is kept as ``build_hardcoded_prompt`` for
backwards compatibility with tests that don't set up an agents/ dir.
"""
from __future__ import annotations

import json
import re
from typing import Any

import yaml


_FRONTMATTER = re.compile(r"^---\s*\n(.*?\n)---\s*\n(.*)$", re.DOTALL)


def parse_agent_md(text: str) -> tuple[dict[str, Any], str]:
    """Split an ``agents/<role>.md`` file into (frontmatter dict, body string)."""
    m = _FRONTMATTER.match(text)
    if not m:
        return {}, text
    fm = yaml.safe_load(m.group(1)) or {}
    body = m.group(2).strip() + "\n"
    return fm, body


# Per-type JSON schema hints. Same set as Phase 2; these go in the user
# message so the model knows the exact JSON shape to return.
SCHEMA_HINTS: dict[str, str] = {
    "ResearchBrief": (
        '{"summary": "<≤500 tokens>", '
        '"key_findings": ["<bullet>"], '
        '"sources": [], '
        '"open_questions": []}'
    ),
    "PatchSummary": (
        '{"files_changed": [], "diff_stat": {}, '
        '"rationale": "<≤300 tokens>", "test_targets": []}'
    ),
    "TestResult": (
        '{"tests_run": 0, "passed": 0, "failed": [], '
        '"runtime_seconds": 0.0, '
        '"metrics": {"<name>": "<number|string|boolean>"}, '
        '"artifact_files": []}'
    ),
    "ProgressLogEntry": (
        '{"summary": "<one paragraph>", "notes": []}'
    ),
    "FailureReport": (
        '{"failure_mode": "logic_error", "error_message": "<text>", '
        '"retry_count": 0, "recommended_action": "retry"}'
    ),
}


def primary_artifact_type(task: dict[str, Any]) -> str:
    output_types = task.get("output_artifact_types") or ["ProgressLogEntry"]
    return output_types[0]


def contract_types(frontmatter: dict[str, Any]) -> list[str]:
    """Allowed output types per the agent's frontmatter contract.

    The framework writes ``ProgressLogEntry`` itself on submit, so a pod
    is allowed to return any type the agent's contract lists *except*
    ProgressLogEntry. We keep ProgressLogEntry in the returned list so
    callers can still ask "is this a known type for this role?"
    """
    raw = frontmatter.get("output_artifact_contract") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(x) for x in raw]


def build_pod_prompt(
    *,
    agent_md_text: str,
    task: dict[str, Any],
    rolling_summary: str = "",
    input_artifacts: list[dict[str, Any]] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Build (system, user, frontmatter) for one pod task.

    System prompt = the agent .md body verbatim (cached at the call
    site via ``cache_control: ephemeral``).
    User prompt   = the per-task variables + a JSON-shape instruction.
    """
    frontmatter, body = parse_agent_md(agent_md_text)
    system = body.rstrip() + "\n"

    primary = primary_artifact_type(task)
    schema_hint = SCHEMA_HINTS.get(primary, SCHEMA_HINTS["ProgressLogEntry"])

    parts: list[str] = []
    parts.append(f"Task ID: {task.get('task_id', '?')}")
    if task.get("working_dir"):
        parts.append(f"Working directory: {task['working_dir']}")
    parts.append(f"\nGoal:\n{task.get('goal_text', '')}\n")

    if rolling_summary.strip():
        parts.append("Rolling summary (current run state):")
        parts.append("------")
        parts.append(rolling_summary.strip())
        parts.append("------\n")

    inputs = input_artifacts or []
    if inputs:
        parts.append("Input artifacts:")
        for a in inputs:
            parts.append(
                f"- {a['artifact_id']} ({a['artifact_type']}):\n"
                + json.dumps(a.get("content", {}), indent=2)
            )
        parts.append("")

    allowed = frontmatter.get("allowed_tools") or []
    if allowed:
        parts.append(
            "You have tools available (write_file, read_file, bash) for any "
            "filesystem or shell work this task requires. Paths are sandboxed "
            f"to the working directory. Use them as needed, then once the work "
            f"is done emit your final {primary} JSON artifact as a single "
            "text response (no prose around it, no code fences)."
        )
        parts.append("")
    parts.append(f"Produce one artifact of type {primary}.")
    parts.append("Output exactly this JSON shape (no prose, no code fences):")
    parts.append(schema_hint)

    user = "\n".join(parts)
    return system, user, frontmatter


# --------------- Phase 2 fallback (kept for tests) -------------------

def build_hardcoded_prompt(task: dict[str, Any]) -> tuple[str, str]:
    role = task.get("agent_role") or "development"
    primary = primary_artifact_type(task)
    schema = SCHEMA_HINTS.get(primary, SCHEMA_HINTS["ProgressLogEntry"])
    system = (
        f"You are a {role} subagent in a multi-agent orchestration framework.\n"
        "You receive one task at a time and produce one structured artifact in response.\n"
        "Reply with a single JSON object only — no surrounding prose, no code fences.\n"
        "Be concise; the framework caps your output budget."
    )
    user = (
        f"Task ID: {task.get('task_id', '?')}\n"
        f"Goal:\n{task.get('goal_text', '')}\n\n"
        f"Produce one artifact of type {primary}.\n"
        "Output exactly this JSON shape:\n"
        f"{schema}\n"
    )
    return system, user


# Back-compat alias used by Phase 2 tests.
build_prompt = build_hardcoded_prompt


def _extract_balanced_json(text: str) -> str | None:
    """Find the first balanced ``{ ... }`` substring and return it.

    Walks the string with a brace counter that respects double-quoted
    strings (and their backslash escapes) so a ``{`` inside a string
    doesn't throw off the count. Returns ``None`` if no top-level
    object is found.
    """
    start = -1
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    return text[start:i + 1]
    return None


def parse_artifact_content(text: str) -> Any:
    """Parse the model's reply as JSON; tolerate code fences and prose.

    Models sometimes emit prose around the JSON ("Here you go:\\n{...}\\n
    Done!"). We try strict JSON first (after fence stripping); on failure
    we extract the first balanced top-level object and try that. Only if
    both fail do we wrap the raw text so the user can see what came back.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    embedded = _extract_balanced_json(text)
    if embedded is not None:
        try:
            return json.loads(embedded)
        except json.JSONDecodeError:
            pass
    return {"raw_text": text, "_parse_error": True}
