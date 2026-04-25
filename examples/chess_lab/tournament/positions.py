from __future__ import annotations

from pathlib import Path

import chess

START_FEN = chess.STARTING_FEN

DEFAULT_DEV_POSITIONS = [
    START_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3",
    "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 2 4",
    "r3k2r/pppq1ppp/2npbn2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w kq - 4 8",
]

DEFAULT_HELDOUT_POSITIONS = [
    START_FEN,
    "rnbq1rk1/ppp2ppp/3bpn2/3p4/3P4/2NBPN2/PPP2PPP/R1BQ1RK1 w - - 2 7",
    "r1bq1rk1/ppp2ppp/2nbpn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 2 7",
    "2kr3r/ppp2ppp/2n1bn2/3qp3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 2 9",
]


def load_positions(path: str | Path | None, heldout: bool = False) -> list[str]:
    if path and Path(path).exists():
        return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    return DEFAULT_HELDOUT_POSITIONS if heldout else DEFAULT_DEV_POSITIONS

