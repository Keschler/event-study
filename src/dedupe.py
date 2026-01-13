from datetime import datetime, timezone
from typing import Any


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)


def merge_dedupe(
    existing_posts: list[dict[str, Any]],
    new_posts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for post in existing_posts + new_posts:
        post_id = str(post.get("id", ""))
        if not post_id:
            continue
        merged[post_id] = post

    sorted_posts = sorted(
        merged.values(),
        key=lambda item: _parse_datetime(
            item.get("created_at") or item.get("date") or ""
        ),
    )
    return sorted_posts
