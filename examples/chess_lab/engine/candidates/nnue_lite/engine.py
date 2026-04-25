from __future__ import annotations

import math
import time

import chess

from engine.common.interface import ChessEngine, safe_legal_move


class NNUELiteEngine(ChessEngine):
    name = "nnue_lite_value"

    weights = {
        chess.PAWN: 0.9,
        chess.KNIGHT: 3.1,
        chess.BISHOP: 3.2,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.1,
        chess.KING: 0.0,
    }

    def select_move(self, position: chess.Board, time_budget_ms: int = 200) -> chess.Move:
        deadline = time.perf_counter() + max(time_budget_ms, 20) / 1000
        best_move: chess.Move | None = None
        best_score = -math.inf
        for move in self._ordered(position):
            if time.perf_counter() >= deadline:
                break
            position.push(move)
            score = -self._one_ply_value(position)
            position.pop()
            if score > best_score:
                best_score = score
                best_move = move
        return safe_legal_move(position, best_move)

    def _one_ply_value(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -999.0
        value = 0.0
        for piece_type, weight in self.weights.items():
            value += len(board.pieces(piece_type, chess.WHITE)) * weight
            value -= len(board.pieces(piece_type, chess.BLACK)) * weight
        value += self._king_pressure(board)
        return value if board.turn == chess.WHITE else -value

    def _king_pressure(self, board: chess.Board) -> float:
        score = 0.0
        for color, sign in [(chess.WHITE, 1.0), (chess.BLACK, -1.0)]:
            king = board.king(not color)
            if king is None:
                continue
            attackers = board.attackers(color, king)
            score += sign * 0.15 * len(attackers)
        return score

    def _ordered(self, board: chess.Board) -> list[chess.Move]:
        return sorted(list(board.legal_moves), key=lambda m: board.is_capture(m) + board.gives_check(m), reverse=True)

