import json
from pathlib import Path


_REQUIRED_KEYS = [
    ("scrape", "username"),
    ("scrape", "max_posts"),
    ("paths", "raw_dir"),
    ("paths", "processed_dir"),
    ("paths", "outputs_dir"),
    ("market", "benchmark"),
    ("market", "symbols"),
    ("event", "event_window"),
    ("event", "estimation_window"),
    ("event", "market_tz"),
]


def _get_nested(config: dict, keys: tuple[str, ...]):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    missing = []
    for key_path in _REQUIRED_KEYS:
        value = _get_nested(config, key_path)
        if value is None:
            missing.append(".".join(key_path))

    if missing:
        raise ValueError(f"Config missing required keys: {', '.join(missing)}")

    return config
