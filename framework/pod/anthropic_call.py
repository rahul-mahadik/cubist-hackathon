"""Make an Anthropic API call and meter it.

The caller is injected so tests can drop in a fake Anthropic client that
returns canned responses (or raises) without touching the network.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

log = logging.getLogger(__name__)


@dataclass
class CallResult:
    text: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cost_usd: float
    duration_seconds: float
    model: str
    raw_stop_reason: str | None = None


class AnthropicLike(Protocol):
    """Minimal duck-typed interface we depend on. ``anthropic.Anthropic``
    satisfies it; tests use a small stand-in."""
    @property
    def messages(self) -> Any: ...


def compute_cost(model: str, usage: Any, pricing: dict[str, dict[str, float]]) -> float:
    p = pricing.get(model)
    if p is None:
        return 0.0
    in_rate = p["input"]
    out_rate = p["output"]
    in_tok = getattr(usage, "input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return (
        in_tok        / 1_000_000 * in_rate
        + out_tok     / 1_000_000 * out_rate
        + cache_read  / 1_000_000 * in_rate * 0.1
        + cache_create / 1_000_000 * in_rate * 1.25
    )


def call_messages(
    client: AnthropicLike,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    pricing: dict[str, dict[str, float]],
) -> CallResult:
    """One Anthropic Messages API call, metered.

    The system prompt is wrapped in a ``cache_control: ephemeral`` block so
    the agent role description gets cached across tasks of the same role.
    Retries on transient 429/5xx are handled by the SDK (we set
    ``max_retries`` when building the client).
    """
    t0 = time.monotonic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user}],
    )
    dt = time.monotonic() - t0

    text_parts = []
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
    text = "".join(text_parts)

    usage = response.usage
    return CallResult(
        text=text,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        cost_usd=compute_cost(model, usage, pricing),
        duration_seconds=dt,
        model=model,
        raw_stop_reason=getattr(response, "stop_reason", None),
    )


def call_messages_agentic(
    client: AnthropicLike,
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    pricing: dict[str, dict[str, float]],
    tools: list[dict[str, Any]],
    tool_handler: Callable[[str, dict[str, Any]], dict[str, Any]],
    max_iterations: int = 12,
) -> CallResult:
    """Agentic Messages loop with tool use.

    Sends ``user`` once, then loops: if the model emits ``tool_use`` blocks,
    we run them via ``tool_handler``, append a ``tool_result`` message, and
    call again. Stops when the model returns no tool_use blocks (typically
    ``stop_reason='end_turn'``) or when ``max_iterations`` is hit.

    Token counts and cost are aggregated across rounds so the caller and
    budget ledger see the full picture.
    """
    t0 = time.monotonic()
    messages: list[dict[str, Any]] = [{"role": "user", "content": user}]

    agg_in = 0
    agg_out = 0
    agg_cache_read = 0
    agg_cache_create = 0
    agg_cost = 0.0
    final_text = ""
    last_stop = None
    exhausted_with_tool_result = False

    for iteration in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            tools=tools,
            messages=messages,
        )
        usage = response.usage
        agg_in           += getattr(usage, "input_tokens", 0) or 0
        agg_out          += getattr(usage, "output_tokens", 0) or 0
        agg_cache_read   += getattr(usage, "cache_read_input_tokens", 0) or 0
        agg_cache_create += getattr(usage, "cache_creation_input_tokens", 0) or 0
        agg_cost         += compute_cost(model, usage, pricing)
        last_stop = getattr(response, "stop_reason", None)

        text_parts: list[str] = []
        tool_uses: list[Any] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                tool_uses.append(block)
        final_text = "".join(text_parts)

        if not tool_uses:
            break

        # Append the assistant turn (must include all blocks verbatim).
        assistant_blocks: list[dict[str, Any]] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                assistant_blocks.append(
                    {"type": "text", "text": getattr(block, "text", "")}
                )
            elif btype == "tool_use":
                assistant_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        messages.append({"role": "assistant", "content": assistant_blocks})

        # Execute each tool_use and build a single tool_result message.
        tool_results: list[dict[str, Any]] = []
        for tu in tool_uses:
            tool_name = getattr(tu, "name", "")
            tool_input = getattr(tu, "input", {}) or {}
            try:
                result = tool_handler(tool_name, dict(tool_input))
            except Exception as e:
                result = {"ok": False, "error": f"handler raised: {e}"}
            log.info(
                "tool %s -> ok=%s (iter=%d)",
                tool_name, result.get("ok"), iteration,
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result),
                "is_error": not result.get("ok", False),
            })
        messages.append({"role": "user", "content": tool_results})
        exhausted_with_tool_result = iteration == max_iterations - 1

    if exhausted_with_tool_result:
        messages.append({
            "role": "user",
            "content": (
                "You have reached the tool-use limit. Based only on the "
                "available tool results, output the final artifact JSON now. "
                "Do not use prose, code fences, or more tool calls."
            ),
        })
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=messages,
        )
        usage = response.usage
        agg_in           += getattr(usage, "input_tokens", 0) or 0
        agg_out          += getattr(usage, "output_tokens", 0) or 0
        agg_cache_read   += getattr(usage, "cache_read_input_tokens", 0) or 0
        agg_cache_create += getattr(usage, "cache_creation_input_tokens", 0) or 0
        agg_cost         += compute_cost(model, usage, pricing)
        last_stop = getattr(response, "stop_reason", None)
        final_text = "".join(
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        )

    dt = time.monotonic() - t0
    return CallResult(
        text=final_text,
        input_tokens=agg_in,
        output_tokens=agg_out,
        cache_read_input_tokens=agg_cache_read,
        cache_creation_input_tokens=agg_cache_create,
        cost_usd=agg_cost,
        duration_seconds=dt,
        model=model,
        raw_stop_reason=last_stop,
    )


def build_anthropic_client(api_key: str, *, max_retries: int = 3):
    """Build a real anthropic.Anthropic client. Imported lazily so unit
    tests don't need the SDK at import time."""
    import anthropic
    return anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
