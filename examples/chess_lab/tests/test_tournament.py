from __future__ import annotations

from tournament.runner import evaluate_engine


def test_evaluate_engine_writes_summary(tmp_path) -> None:
    summary = evaluate_engine("alphabeta", move_budget_ms=20, games_per_position=1, output_dir=tmp_path)
    assert summary["games"] > 0
    assert 0.0 <= summary["win_rate"] <= 1.0
    assert summary["artifact_files"]

