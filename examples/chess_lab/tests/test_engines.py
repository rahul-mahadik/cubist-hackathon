from __future__ import annotations

import chess
import pytest

from engine.common.registry import create_engine


@pytest.mark.parametrize("kind", ["alphabeta", "mcts", "nnue_lite", "policy_guided"])
def test_candidate_returns_legal_move(kind: str) -> None:
    board = chess.Board()
    engine = create_engine(kind)
    move = engine.select_move(board, time_budget_ms=25)
    assert move in board.legal_moves


@pytest.mark.parametrize(
    "fen",
    [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "8/P7/8/8/8/8/7p/4K2k w - - 0 1",
    ],
)
def test_special_positions_do_not_crash(fen: str) -> None:
    board = chess.Board(fen)
    for kind in ["alphabeta", "mcts", "nnue_lite", "policy_guided"]:
        move = create_engine(kind).select_move(board.copy(stack=False), time_budget_ms=25)
        assert move in board.legal_moves

