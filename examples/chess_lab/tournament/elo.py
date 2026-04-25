from __future__ import annotations

import math


def approximate_elo_delta(score_rate: float) -> float:
    score_rate = min(0.99, max(0.01, score_rate))
    return -400 * math.log10(1 / score_rate - 1)

