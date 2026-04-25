from __future__ import annotations

from abc import ABC, abstractmethod

import chess


class ChessEngine(ABC):
    name: str = "engine"

    @abstractmethod
    def select_move(self, position: chess.Board, time_budget_ms: int = 200) -> chess.Move:
        """Return a legal move for the given board."""


def safe_legal_move(board: chess.Board, move: chess.Move | None) -> chess.Move:
    if move in board.legal_moves:
        return move
    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("No legal moves available")
    return legal[0]

