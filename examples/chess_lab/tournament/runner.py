from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.pgn

from engine.common.baseline import RandomBaseline
from engine.common.registry import create_engine
from tournament.elo import approximate_elo_delta
from tournament.logging_utils import append_jsonl
from tournament.positions import load_positions


@dataclass
class GameResult:
    white: str
    black: str
    result: str
    plies: int
    illegal_moves: int
    crashes: int
    avg_move_latency_ms: float
    start_fen: str


def play_game(white_engine, black_engine, start_fen: str, move_budget_ms: int = 200, max_plies: int = 120) -> tuple[GameResult, chess.pgn.Game]:
    board = chess.Board(start_fen)
    game = chess.pgn.Game()
    game.headers["White"] = white_engine.name
    game.headers["Black"] = black_engine.name
    game.setup(board)
    node = game
    latencies: list[float] = []
    illegal = 0
    crashes = 0
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        engine = white_engine if board.turn == chess.WHITE else black_engine
        started = time.perf_counter()
        try:
            move = engine.select_move(board.copy(stack=False), move_budget_ms)
        except Exception as exc:
            crashes += 1
            append_jsonl(
                "logs/engine_checks.jsonl",
                {
                    "event": "engine_exception",
                    "engine": engine.name,
                    "fen": board.fen(),
                    "error": str(exc),
                },
            )
            move = next(iter(board.legal_moves), None)
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        if move not in board.legal_moves:
            illegal += 1
            append_jsonl(
                "logs/engine_checks.jsonl",
                {
                    "event": "illegal_move",
                    "engine": engine.name,
                    "fen": board.fen(),
                    "move": str(move),
                },
            )
            move = next(iter(board.legal_moves), None)
        if move is None:
            break
        board.push(move)
        node = node.add_variation(move)
    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
    game.headers["Result"] = result
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    return (
        GameResult(white_engine.name, black_engine.name, result, board.ply(), illegal, crashes, avg_latency, start_fen),
        game,
    )


def score_for(engine_name: str, game: GameResult) -> float:
    if game.result == "1/2-1/2":
        return 0.5
    if game.result == "1-0":
        return 1.0 if game.white == engine_name else 0.0
    if game.result == "0-1":
        return 1.0 if game.black == engine_name else 0.0
    return 0.5


def evaluate_engine(kind: str, positions_path: str | None = None, heldout: bool = False, move_budget_ms: int = 200, games_per_position: int = 2, output_dir: str | Path | None = None) -> dict:
    append_jsonl(
        "logs/eval_runs.jsonl",
        {
            "event": "eval_started",
            "kind": kind,
            "heldout": heldout,
            "move_budget_ms": move_budget_ms,
            "games_per_position": games_per_position,
        },
    )
    engine = create_engine(kind)
    baseline = RandomBaseline()
    positions = load_positions(positions_path, heldout=heldout)
    games: list[GameResult] = []
    pgns: list[str] = []
    for fen in positions:
        pairings = [(engine, baseline), (baseline, engine)] if games_per_position >= 2 else [(engine, baseline)]
        for white, black in pairings:
            result, game = play_game(white, black, fen, move_budget_ms)
            games.append(result)
            pgns.append(str(game))
            append_jsonl(
                "logs/engine_checks.jsonl",
                {
                    "event": "game_completed",
                    "kind": kind,
                    "white": result.white,
                    "black": result.black,
                    "result": result.result,
                    "plies": result.plies,
                    "illegal_moves": result.illegal_moves,
                    "crashes": result.crashes,
                    "avg_move_latency_ms": result.avg_move_latency_ms,
                    "start_fen": result.start_fen,
                },
            )
    scores = [score_for(engine.name, game) for game in games]
    win_rate = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "engine": engine.name,
        "kind": kind,
        "heldout": heldout,
        "games": len(games),
        "win_rate": win_rate,
        "elo_delta": approximate_elo_delta(win_rate),
        "illegal_moves": sum(g.illegal_moves for g in games),
        "crashes": sum(g.crashes for g in games),
        "avg_move_latency_ms": sum(g.avg_move_latency_ms for g in games) / len(games) if games else 0.0,
        "results": [asdict(g) for g in games],
    }
    import os

    artifacts_root = Path(os.getenv("CHESS_LAB_ARTIFACTS_DIR", "artifacts"))
    out = Path(output_dir) if output_dir else artifacts_root / "evals"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{kind}_{'heldout' if heldout else 'dev'}_{int(time.time())}"
    (out / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / f"{stem}.pgn").write_text("\n\n".join(pgns), encoding="utf-8")
    summary["artifact_files"] = [str(out / f"{stem}.json"), str(out / f"{stem}.pgn")]
    append_jsonl(
        "logs/eval_runs.jsonl",
        {
            "event": "eval_completed",
            "kind": kind,
            "heldout": heldout,
            "games": summary["games"],
            "win_rate": summary["win_rate"],
            "elo_delta": summary["elo_delta"],
            "illegal_moves": summary["illegal_moves"],
            "crashes": summary["crashes"],
            "avg_move_latency_ms": summary["avg_move_latency_ms"],
            "artifact_files": summary["artifact_files"],
        },
    )
    return summary


def round_robin(kinds: list[str], positions_path: str | None = None, move_budget_ms: int = 200) -> dict:
    positions = load_positions(positions_path)
    engines = [(kind, create_engine(kind)) for kind in kinds]
    games: list[GameResult] = []
    for fen in positions:
        for i, (_, white) in enumerate(engines):
            for j, (_, black) in enumerate(engines):
                if i >= j:
                    continue
                result, _ = play_game(white, black, fen, move_budget_ms)
                games.append(result)
                result, _ = play_game(black, white, fen, move_budget_ms)
                games.append(result)
    return {"games": [asdict(g) for g in games]}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["alphabeta", "mcts", "nnue_lite", "policy_guided"])
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--move-budget-ms", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(evaluate_engine(args.kind, heldout=args.heldout, move_budget_ms=args.move_budget_ms), indent=2))
