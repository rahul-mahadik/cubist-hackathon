from __future__ import annotations

import math
import time

import chess

from engine.common.eval import evaluate_handcrafted
from engine.common.interface import ChessEngine, safe_legal_move


class AlphaBetaEngine(ChessEngine):
    name = "classical_alphabeta"

    def select_move(self, position: chess.Board, time_budget_ms: int = 200) -> chess.Move:
        deadline = time.perf_counter() + max(time_budget_ms, 20) / 1000
        best_move: chess.Move | None = None
        best_score = -math.inf
        moves = self._ordered_moves(position)
        depth = 2 if len(moves) > 16 else 3
        for move in moves:
            if time.perf_counter() >= deadline:
                break
            position.push(move)
            score = -self._search(position, depth - 1, -math.inf, math.inf, deadline)
            position.pop()
            if score > best_score:
                best_score = score
                best_move = move
        return safe_legal_move(position, best_move)

    def _search(self, board: chess.Board, depth: int, alpha: float, beta: float, deadline: float) -> float:
        if depth == 0 or board.is_game_over() or time.perf_counter() >= deadline:
            return self._relative_eval(board)
        for move in self._ordered_moves(board):
            board.push(move)
            score = -self._search(board, depth - 1, -beta, -alpha, deadline)
            board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def _relative_eval(self, board: chess.Board) -> int:
        score = evaluate_handcrafted(board)
        return score if board.turn == chess.WHITE else -score

    def _ordered_moves(self, board: chess.Board) -> list[chess.Move]:
        moves = list(board.legal_moves)
        return sorted(moves, key=lambda m: board.is_capture(m) * 10 + board.gives_check(m), reverse=True)

