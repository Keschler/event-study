"""Module for writing immutable raw data dumps."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any


def write_raw_dump(
    raw_dir: str,
    username: str,
    posts: List[Dict[str, Any]]
) -> Path:
    """
    Write raw posts to an immutable timestamped file.
    
    Each scraping run creates a new file with a timestamp, ensuring
    immutable raw data history.
    
    Args:
        raw_dir: Directory to write raw data files
        username: Username being scraped
        posts: List of post dictionaries
        
    Returns:
        Path to the created file
    """
    # Create raw directory if it doesn't exist
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp in ISO format (safe for filenames)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    
    # Create filename: YYYY-MM-DDTHH-MM-SSZ_username.json
    filename = f"{timestamp}_{username}.json"
    filepath = raw_path / filename
    
    # Write posts as JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    
    print(f"Wrote {len(posts)} posts to raw dump: {filepath}")
    return filepath


def list_raw_dumps(raw_dir: str, username: str) -> List[Path]:
    """
    List all raw dump files for a given username, sorted by timestamp.
    
    Args:
        raw_dir: Directory containing raw data files
        username: Username to filter by
        
    Returns:
        List of Path objects sorted by timestamp (oldest first)
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return []
    
    # Find all files matching the pattern *_username.json
    pattern = f"*_{username}.json"
    files = list(raw_path.glob(pattern))
    
    # Sort by filename (which includes timestamp)
    files.sort()
    
    return files


def load_raw_dump(filepath: Path) -> List[Dict[str, Any]]:
    """
    Load posts from a raw dump file.
    
    Args:
        filepath: Path to the raw dump file
        
    Returns:
        List of post dictionaries
    """
    with open(filepath, "r", encoding="utf-8") as f:
        posts = json.load(f)
    
    return posts
