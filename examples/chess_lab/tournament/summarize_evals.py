from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_eval(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = {"kind", "games", "win_rate", "elo_delta"}
    if not required.issubset(data):
        return None
    data["_path"] = str(path)
    return data


def load_evals(evals_dir: Path) -> list[dict[str, Any]]:
    return [
        data
        for path in sorted(evals_dir.glob("*.json"))
        if (data := _load_eval(path)) is not None
    ]


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| kind | heldout | games | win_rate | elo_delta | illegal | crashes | avg_latency_ms | file |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {kind} | {heldout} | {games} | {win_rate:.3f} | {elo_delta:.1f} | "
            "{illegal_moves} | {crashes} | {avg_move_latency_ms:.1f} | {path} |".format(
                kind=row["kind"],
                heldout=str(bool(row.get("heldout", False))).lower(),
                games=int(row.get("games", 0)),
                win_rate=float(row.get("win_rate", 0.0)),
                elo_delta=float(row.get("elo_delta", 0.0)),
                illegal_moves=int(row.get("illegal_moves", 0)),
                crashes=int(row.get("crashes", 0)),
                avg_move_latency_ms=float(row.get("avg_move_latency_ms", 0.0)),
                path=row["_path"],
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evals-dir", default="artifacts/evals")
    args = parser.parse_args()
    rows = load_evals(Path(args.evals_dir))
    print(render_markdown(rows) if rows else "No eval JSON files found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
