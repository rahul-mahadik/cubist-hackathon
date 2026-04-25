from __future__ import annotations

import json

from tournament.summarize_evals import load_evals, render_markdown
from tournament.runner import evaluate_engine


def test_evaluate_engine_writes_summary(tmp_path) -> None:
    summary = evaluate_engine("alphabeta", move_budget_ms=20, games_per_position=1, output_dir=tmp_path)
    assert summary["games"] > 0
    assert 0.0 <= summary["win_rate"] <= 1.0
    assert "elo_delta" in summary
    assert summary["artifact_files"]


def test_summarize_evals_includes_elo(tmp_path) -> None:
    (tmp_path / "sample.json").write_text(
        json.dumps({
            "kind": "alphabeta",
            "heldout": False,
            "games": 4,
            "win_rate": 0.75,
            "elo_delta": 190.8,
            "illegal_moves": 0,
            "crashes": 0,
            "avg_move_latency_ms": 12.3,
        }),
        encoding="utf-8",
    )
    table = render_markdown(load_evals(tmp_path))
    assert "elo_delta" in table
    assert "190.8" in table
