"""Tests for the pod tool-use loop (write_file/read_file/bash) and the
agentic Anthropic call wrapper. Covers the v1.x patch that wired tool use
into the pod after the v1 release shipped without it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from framework.pod.anthropic_call import call_messages_agentic
from framework.pod.tools import build_tools


# ----------------------------- build_tools ---------------------------

def test_build_tools_returns_empty_when_no_working_dir(tmp_path):
    schemas, handler = build_tools(
        {"allowed_tools": ["filesystem_write"]}, working_dir=None,
    )
    assert schemas == []
    assert handler is None


def test_build_tools_returns_empty_when_working_dir_does_not_exist(tmp_path):
    schemas, handler = build_tools(
        {"allowed_tools": ["filesystem_write"]},
        working_dir=str(tmp_path / "ghost"),
    )
    assert schemas == []
    assert handler is None


def test_build_tools_maps_allowed_tools_to_tool_names(tmp_path):
    schemas, _ = build_tools(
        {"allowed_tools": ["filesystem_read", "filesystem_write", "bash"]},
        working_dir=str(tmp_path),
    )
    names = sorted(s["name"] for s in schemas)
    assert names == ["bash", "read_file", "write_file"]


def test_build_tools_omits_write_for_read_only_role(tmp_path):
    """Testing role's allowed_tools excludes filesystem_write — pods
    must not get a write_file tool, enforcing the read-only contract."""
    schemas, _ = build_tools(
        {"allowed_tools": ["filesystem_read", "bash"]},
        working_dir=str(tmp_path),
    )
    names = {s["name"] for s in schemas}
    assert "write_file" not in names
    assert names == {"read_file", "bash"}


def test_build_tools_handler_writes_file_inside_working_dir(tmp_path):
    _, handler = build_tools(
        {"allowed_tools": ["filesystem_write"]}, working_dir=str(tmp_path),
    )
    result = handler("write_file", {"path": "hello.py", "content": "print('hi')\n"})
    assert result["ok"] is True
    assert (tmp_path / "hello.py").read_text() == "print('hi')\n"


def test_build_tools_handler_creates_parent_dirs(tmp_path):
    _, handler = build_tools(
        {"allowed_tools": ["filesystem_write"]}, working_dir=str(tmp_path),
    )
    result = handler("write_file", {"path": "a/b/c.txt", "content": "x"})
    assert result["ok"] is True
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "x"


def test_build_tools_handler_rejects_path_outside_working_dir(tmp_path):
    _, handler = build_tools(
        {"allowed_tools": ["filesystem_write"]}, working_dir=str(tmp_path),
    )
    result = handler("write_file", {"path": "../escape.txt", "content": "x"})
    assert result["ok"] is False
    assert "outside" in result["error"]


def test_build_tools_handler_rejects_absolute_path_outside(tmp_path):
    _, handler = build_tools(
        {"allowed_tools": ["filesystem_write"]}, working_dir=str(tmp_path),
    )
    result = handler("write_file", {"path": "/etc/passwd", "content": "x"})
    assert result["ok"] is False
    assert "outside" in result["error"]


def test_build_tools_handler_reads_file(tmp_path):
    (tmp_path / "f.txt").write_text("hello\n")
    _, handler = build_tools(
        {"allowed_tools": ["filesystem_read"]}, working_dir=str(tmp_path),
    )
    result = handler("read_file", {"path": "f.txt"})
    assert result["ok"] is True
    assert result["content"] == "hello\n"


def test_build_tools_handler_runs_bash_in_working_dir(tmp_path):
    (tmp_path / "marker").write_text("found")
    _, handler = build_tools(
        {"allowed_tools": ["bash"]}, working_dir=str(tmp_path),
    )
    result = handler("bash", {"command": "cat marker"})
    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert "found" in result["stdout"]


def test_build_tools_handler_reports_nonzero_exit(tmp_path):
    _, handler = build_tools(
        {"allowed_tools": ["bash"]}, working_dir=str(tmp_path),
    )
    result = handler("bash", {"command": "exit 3"})
    assert result["ok"] is False
    assert result["exit_code"] == 3


def test_build_tools_handler_unknown_tool(tmp_path):
    _, handler = build_tools(
        {"allowed_tools": ["filesystem_write"]}, working_dir=str(tmp_path),
    )
    result = handler("nonsense", {})
    assert result["ok"] is False
    assert "unknown" in result["error"]


# ----------------------- call_messages_agentic -----------------------

@dataclass
class _Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class _Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class _Resp:
    content: list
    usage: _Usage
    stop_reason: str = "end_turn"


class _ScriptedClient:
    """Returns canned responses in order. Matches the duck-typed
    ``client.messages.create(...)`` interface ``call_messages_agentic``
    uses.
    """
    def __init__(self, responses: list[_Resp]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("scripted client out of responses")
        return self._responses.pop(0)


_PRICING = {"m": {"input": 1.0, "output": 5.0}}


def test_agentic_loop_executes_tool_and_aggregates_metering():
    """Round 1 emits tool_use → handler runs → round 2 returns final text."""
    client = _ScriptedClient([
        _Resp(
            content=[
                _Block(type="text", text="thinking…"),
                _Block(type="tool_use", id="tu_1", name="write_file",
                       input={"path": "f.py", "content": "x = 1\n"}),
            ],
            usage=_Usage(input_tokens=100, output_tokens=20),
            stop_reason="tool_use",
        ),
        _Resp(
            content=[_Block(type="text", text='{"files_changed": ["f.py"]}')],
            usage=_Usage(input_tokens=150, output_tokens=10),
        ),
    ])

    invocations: list[tuple[str, dict]] = []
    def handler(name, args):
        invocations.append((name, args))
        return {"ok": True, "bytes_written": len(args.get("content", ""))}

    result = call_messages_agentic(
        client, model="m", system="sys", user="goal",
        max_tokens=512, pricing=_PRICING,
        tools=[{"name": "write_file"}], tool_handler=handler,
    )

    assert result.text == '{"files_changed": ["f.py"]}'
    assert result.input_tokens == 250
    assert result.output_tokens == 30
    assert invocations == [("write_file", {"path": "f.py", "content": "x = 1\n"})]
    assert len(client.calls) == 2
    # Round 2 must have included the tool_result message after the assistant turn.
    round2 = client.calls[1]
    assert round2["messages"][-1]["role"] == "user"
    assert round2["messages"][-1]["content"][0]["type"] == "tool_result"
    assert round2["messages"][-1]["content"][0]["tool_use_id"] == "tu_1"


def test_agentic_loop_terminates_when_no_tool_use():
    """Single round, model just emits text — agentic loop must return immediately."""
    client = _ScriptedClient([
        _Resp(
            content=[_Block(type="text", text='{"ok": true}')],
            usage=_Usage(input_tokens=50, output_tokens=5),
        ),
    ])
    result = call_messages_agentic(
        client, model="m", system="s", user="u", max_tokens=64,
        pricing=_PRICING, tools=[], tool_handler=lambda *a, **k: {},
    )
    assert result.text == '{"ok": true}'
    assert len(client.calls) == 1


def test_agentic_loop_caps_iterations():
    """A misbehaving model that always emits tool_use must not loop forever."""
    def make_response():
        return _Resp(
            content=[_Block(type="tool_use", id="x", name="bash",
                            input={"command": "echo"})],
            usage=_Usage(input_tokens=10, output_tokens=5),
            stop_reason="tool_use",
        )
    client = _ScriptedClient([
        *[make_response() for _ in range(5)],
        _Resp(
            content=[_Block(type="text", text='{"tests_run": 1, "passed": 1}')],
            usage=_Usage(input_tokens=30, output_tokens=7),
        ),
    ])
    result = call_messages_agentic(
        client, model="m", system="s", user="u", max_tokens=64,
        pricing=_PRICING, tools=[{"name": "bash"}],
        tool_handler=lambda *a, **k: {"ok": True, "stdout": "", "stderr": "", "exit_code": 0},
        max_iterations=5,
    )
    # We made max_iterations tool calls, then one final no-tool synthesis call.
    assert len(client.calls) == 6
    assert "tools" not in client.calls[-1]
    assert result.text == '{"tests_run": 1, "passed": 1}'
    # Aggregated metering is sum across rounds.
    assert result.input_tokens == 80  # 10 * 5 + 30
    assert result.output_tokens == 32  # 5 * 5 + 7


def test_agentic_loop_marks_handler_failure_as_tool_error():
    """When the handler returns ok=False, the tool_result is_error=true so
    the model knows to retry or abort."""
    client = _ScriptedClient([
        _Resp(
            content=[_Block(type="tool_use", id="tu_1", name="write_file",
                            input={"path": "../bad", "content": "x"})],
            usage=_Usage(input_tokens=10, output_tokens=5),
            stop_reason="tool_use",
        ),
        _Resp(
            content=[_Block(type="text", text='{"giving up": true}')],
            usage=_Usage(input_tokens=20, output_tokens=5),
        ),
    ])
    result = call_messages_agentic(
        client, model="m", system="s", user="u", max_tokens=64,
        pricing=_PRICING, tools=[{"name": "write_file"}],
        tool_handler=lambda n, a: {"ok": False, "error": "outside working_dir"},
    )
    assert result.text == '{"giving up": true}'
    round2 = client.calls[1]
    tr = round2["messages"][-1]["content"][0]
    assert tr["type"] == "tool_result"
    assert tr["is_error"] is True
