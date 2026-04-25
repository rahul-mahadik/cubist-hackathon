from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import chess

from engine.common.eval import score_for_side
from engine.common.interface import ChessEngine, safe_legal_move


@dataclass
class Node:
    move: chess.Move | None
    parent: "Node | None" = None
    visits: int = 0
    value: float = 0.0
    children: dict[chess.Move, "Node"] = field(default_factory=dict)
    untried: list[chess.Move] = field(default_factory=list)


class MCTSEngine(ChessEngine):
    name = "mcts_puct_lite"

    def select_move(self, position: chess.Board, time_budget_ms: int = 200) -> chess.Move:
        root = Node(move=None, untried=list(position.legal_moves))
        deadline = time.perf_counter() + max(time_budget_ms, 20) / 1000
        while time.perf_counter() < deadline:
            board = position.copy(stack=False)
            node = self._select(root, board)
            value = self._rollout_value(board)
            self._backup(node, value)
        if not root.children:
            return safe_legal_move(position, None)
        best = max(root.children.values(), key=lambda child: child.visits)
        return safe_legal_move(position, best.move)

    def _select(self, node: Node, board: chess.Board) -> Node:
        while not board.is_game_over():
            if node.untried:
                move = self._policy_pick(board, node.untried)
                node.untried.remove(move)
                board.push(move)
                child = Node(move=move, parent=node, untried=list(board.legal_moves))
                node.children[move] = child
                return child
            if not node.children:
                return node
            node = max(node.children.values(), key=lambda c: self._uct(c))
            board.push(node.move)
        return node

    def _uct(self, child: Node) -> float:
        if child.visits == 0:
            return math.inf
        parent_visits = max(1, child.parent.visits if child.parent else 1)
        return child.value / child.visits + 1.4 * math.sqrt(math.log(parent_visits + 1) / child.visits)

    def _policy_pick(self, board: chess.Board, moves: list[chess.Move]) -> chess.Move:
        captures = [m for m in moves if board.is_capture(m)]
        checks = [m for m in moves if board.gives_check(m)]
        return random.choice(checks or captures or moves)

    def _rollout_value(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -1.0
        if board.is_game_over():
            return 0.0
        score = score_for_side(board)
        return max(-1.0, min(1.0, score / 1200))

    def _backup(self, node: Node, value: float) -> None:
        while node:
            node.visits += 1
            node.value += value
            value = -value
            node = node.parent

