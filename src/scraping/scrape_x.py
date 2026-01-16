import asyncio
import random
import time
from datetime import timezone
from typing import Any
from twikit import Client
from twikit.errors import TooManyRequests


async def _run_with_retry(
    task,
    retries: int,
    sleep_s: float,
    *,
    max_wait_s: float = 3600.0,
    max_rate_limit_hits: int = 5,
):
    attempt = 0
    rate_limit_hits = 0

    while True:
        try:
            return await task()

        except TooManyRequests as e:
            rate_limit_hits += 1

            # Twikit setzt e.rate_limit_reset aus 'x-rate-limit-reset' (epoch seconds), wenn vorhanden.
            reset = getattr(e, "rate_limit_reset", None)
            now = time.time()

            if reset is not None:
                wait_s = max(0.0, float(reset) - now) + 5.0  # 5s safety cushion
            else:
                # Fallback: konservativer Backoff, wenn kein Reset bekannt ist
                wait_s = min(max(30.0, sleep_s * (2 ** min(attempt, 6))), max_wait_s)

            # jitter (±10%)
            wait_s *= 0.9 + 0.2 * random.random()
            wait_s = min(wait_s, max_wait_s)

            print(f"[rate-limit] 429 hit; sleeping {wait_s:.1f}s (hit #{rate_limit_hits})")
            await asyncio.sleep(wait_s)

            if rate_limit_hits >= max_rate_limit_hits:
                raise RuntimeError(
                    f"Rate limit hit {rate_limit_hits} times; giving up after {wait_s:.1f}s."
                )

            # Nicht attempt++ für 429 (du willst nicht “schneller sterben”, sondern korrekt warten)
            continue

        except Exception:
            if attempt >= retries:
                raise
            delay = min(sleep_s * (2 ** attempt), 8.0)
            delay *= 0.9 + 0.2 * random.random()
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
    page_size: int,
    page_sleep_s: float,
    max_wait_s: float,
    max_rate_limit_hits: int,
) -> list[dict[str, Any]]:
    client = Client("en-US")
    client.load_cookies(cookies_path)

    user = await _run_with_retry(
        lambda: client.get_user_by_screen_name(username),
        retries,
        sleep_s,
        max_wait_s=max_wait_s,
        max_rate_limit_hits=max_rate_limit_hits,
    )
    user_id = user.id

    posts: list[dict[str, Any]] = []

    result = await _run_with_retry(
        lambda: client.get_user_tweets(user_id, "Tweets", count=min(page_size, max_posts)),
        retries,
        sleep_s,
        max_wait_s=max_wait_s,
        max_rate_limit_hits=max_rate_limit_hits,
    )

    while True:
        for tweet in result:
            if hasattr(tweet, "retweeted_tweet") or hasattr(tweet, "retweeted_status"):
                continue
            posts.append(_tweet_to_dict(tweet))
            if len(posts) >= max_posts:
                break
            print(f"[page-done] scraped_so_far={len(posts)}")
        if len(posts) >= max_posts:
            break
        try:
            result = await _run_with_retry(
                lambda: result.next(),
                retries,
                sleep_s,
                max_wait_s=max_wait_s,
                max_rate_limit_hits=max_rate_limit_hits,
            )
        except Exception:
            break
        await asyncio.sleep(page_sleep_s)

    return posts


def scrape_user_posts(
    username: str,
    max_posts: int,
    cookies_path: str,
    sleep_s: float = 1.0,
    retries: int = 4,
    page_size: int = 40,
    page_sleep_s: float | None = None,
    max_wait_s: float = 3600.0,
    max_rate_limit_hits: int = 5,
) -> list[dict[str, Any]]:
    if page_sleep_s is None:
        page_sleep_s = sleep_s
    return asyncio.run(
        _scrape_async(
            username,
            max_posts,
            cookies_path,
            sleep_s,
            retries,
            page_size,
            page_sleep_s,
            max_wait_s,
            max_rate_limit_hits,
        )
    )
