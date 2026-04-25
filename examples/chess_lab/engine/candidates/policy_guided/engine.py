from __future__ import annotations

import math
import time

import chess

from engine.common.eval import evaluate_handcrafted
from engine.common.interface import ChessEngine, safe_legal_move


class PolicyGuidedEngine(ChessEngine):
    name = "policy_guided_search"

    def select_move(self, position: chess.Board, time_budget_ms: int = 200) -> chess.Move:
        deadline = time.perf_counter() + max(time_budget_ms, 20) / 1000
        top_moves = self._policy_top_k(position, k=8)
        best_move: chess.Move | None = None
        best_score = -math.inf
        for move in top_moves:
            if time.perf_counter() >= deadline:
                break
            position.push(move)
            score = -self._shallow(position)
            position.pop()
            if score > best_score:
                best_score = score
                best_move = move
        return safe_legal_move(position, best_move)

    def _policy_top_k(self, board: chess.Board, k: int) -> list[chess.Move]:
        def prior(move: chess.Move) -> int:
            score = 0
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += 10 * victim.piece_type - attacker.piece_type
            if board.gives_check(move):
                score += 8
            if chess.square_file(move.to_square) in (3, 4) and chess.square_rank(move.to_square) in (2, 3, 4, 5):
                score += 3
            if move.promotion:
                score += 20
            return score

        moves = sorted(list(board.legal_moves), key=prior, reverse=True)
        return moves[:k] or moves

    def _shallow(self, board: chess.Board) -> float:
        if board.is_game_over():
            return self._relative(board)
        best = -math.inf
        for reply in list(board.legal_moves)[:12]:
            board.push(reply)
            best = max(best, -self._relative(board))
            board.pop()
        return best if best != -math.inf else self._relative(board)

    def _relative(self, board: chess.Board) -> int:
        score = evaluate_handcrafted(board)
        return score if board.turn == chess.WHITE else -score

