from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    raw_path = Path(path)
    artifacts_root = Path(os.getenv("CHESS_LAB_ARTIFACTS_DIR", "artifacts"))
    target = raw_path if raw_path.is_absolute() else artifacts_root / raw_path
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {"created_at": datetime.now(timezone.utc).isoformat(), **payload}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
