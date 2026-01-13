import asyncio
from datetime import datetime, timezone
from typing import Any

from twikit import Client


async def _run_with_retry(task, retries: int, sleep_s: float):
    attempt = 0
    while True:
        try:
            return await task()
        except Exception:
            if attempt >= retries:
                raise
            delay = min(sleep_s * (2**attempt), 8.0)
            await asyncio.sleep(delay)
            attempt += 1


def _tweet_to_dict(tweet) -> dict[str, Any]:
    url = f"https://x.com/{tweet.user.screen_name}/status/{tweet.id}"
    dt = getattr(tweet, "created_at_datetime", None)
    if dt is None:
        created_at = getattr(tweet, "created_at", "")
    else:
        created_at = dt.astimezone(timezone.utc).isoformat()
    return {
        "id": str(tweet.id),
        "created_at": created_at,
        "text": tweet.full_text,
        "url": url,
        "user": tweet.user.screen_name,
    }


async def _scrape_async(
    username: str,
    max_posts: int,
    cookies_path: str,
    sleep_s: float,
    retries: int,
) -> list[dict[str, Any]]:
    client = Client("en-US")
    client.load_cookies(cookies_path)

    user = await _run_with_retry(
        lambda: client.get_user_by_screen_name(username), retries, sleep_s
    )
    user_id = user.id

    posts: list[dict[str, Any]] = []

    result = await _run_with_retry(
        lambda: client.get_user_tweets(user_id, "Tweets", count=min(100, max_posts)),
        retries,
        sleep_s,
    )

    while True:
        for tweet in result:
            if hasattr(tweet, "retweeted_tweet") or hasattr(tweet, "retweeted_status"):
                continue
            posts.append(_tweet_to_dict(tweet))
            if len(posts) >= max_posts:
                break
        if len(posts) >= max_posts:
            break
        try:
            result = await _run_with_retry(lambda: result.next(), retries, sleep_s)
        except Exception:
            break
        await asyncio.sleep(sleep_s)

    return posts


def scrape_user_posts(
    username: str,
    max_posts: int,
    cookies_path: str,
    sleep_s: float = 1.0,
    retries: int = 4,
) -> list[dict[str, Any]]:
    return asyncio.run(
        _scrape_async(username, max_posts, cookies_path, sleep_s, retries)
    )
