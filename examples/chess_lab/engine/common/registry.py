from __future__ import annotations

from engine.candidates.alphabeta.engine import AlphaBetaEngine
from engine.candidates.mcts.engine import MCTSEngine
from engine.candidates.nnue_lite.engine import NNUELiteEngine
from engine.candidates.policy_guided.engine import PolicyGuidedEngine


def create_engine(kind: str):
    engines = {
        "alphabeta": AlphaBetaEngine,
        "mcts": MCTSEngine,
        "nnue_lite": NNUELiteEngine,
        "policy_guided": PolicyGuidedEngine,
    }
    if kind not in engines:
        raise KeyError(f"Unknown engine kind: {kind}")
    return engines[kind]()

