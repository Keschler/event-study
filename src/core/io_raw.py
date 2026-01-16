import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_raw_dump(raw_dir: str | Path, username: str, posts: list[dict[str, Any]]) -> Path:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    filename = f"{timestamp}_{username}.json"
    output_path = raw_path / filename

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(posts, handle, ensure_ascii=False, indent=2)

    return output_path
