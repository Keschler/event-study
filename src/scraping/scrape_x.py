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
    min_interval_s: float = 0.0,
    request_state: dict[str, float] | None = None,
    max_rate_limit_hits: int = 5,
    stop_on_rate_limit: bool = False,
):
    attempt = 0
    rate_limit_hits = 0

    while True:
        try:
            if min_interval_s > 0 and request_state is not None:
                last_call = request_state.get("last_call", 0.0)
                wait_for = min_interval_s - (time.time() - last_call)
                if wait_for > 0:
                    await asyncio.sleep(wait_for)
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

            if stop_on_rate_limit:
                print(
                    "[rate-limit] 429 hit; stopping early to avoid long wait "
                    f"(hit #{rate_limit_hits})"
                )
                return None

            print(f"[rate-limit] 429 hit; sleeping {wait_s:.1f}s (hit #{rate_limit_hits})")
            await asyncio.sleep(wait_s)

            # Wenn du ewig in 429 läufst, irgendwann abbrechen statt unendlich zu warten:
            if rate_limit_hits >= max_rate_limit_hits and reset is None:
                raise

            # Nicht attempt++ für 429 (du willst nicht “schneller sterben”, sondern korrekt warten)
            continue

        except Exception:
            if attempt >= retries:
                raise
            delay = min(sleep_s * (2 ** attempt), 8.0)
            delay *= 0.9 + 0.2 * random.random()
            await asyncio.sleep(delay)
            attempt += 1
        finally:
            if request_state is not None:
                request_state["last_call"] = time.time()


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
    *,
    page_sleep_s: float,
    page_size: int,
    min_request_interval_s: float,
    max_rate_limit_hits: int,
    stop_on_rate_limit: bool,
) -> list[dict[str, Any]]:
    client = Client("en-US")
    client.load_cookies(cookies_path)
    request_state: dict[str, float] = {"last_call": 0.0}

    user = await _run_with_retry(
        lambda: client.get_user_by_screen_name(username),
        retries,
        sleep_s,
        min_interval_s=min_request_interval_s,
        request_state=request_state,
        max_rate_limit_hits=max_rate_limit_hits,
        stop_on_rate_limit=stop_on_rate_limit,
    )
    if user is None:
        return []
    user_id = user.id

    posts: list[dict[str, Any]] = []

    result = await _run_with_retry(
        lambda: client.get_user_tweets(user_id, "Tweets", count=min(page_size, max_posts)),
        retries,
        sleep_s,
        min_interval_s=min_request_interval_s,
        request_state=request_state,
        max_rate_limit_hits=max_rate_limit_hits,
        stop_on_rate_limit=stop_on_rate_limit,
    )
    if result is None:
        return posts

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
                min_interval_s=min_request_interval_s,
                request_state=request_state,
                max_rate_limit_hits=max_rate_limit_hits,
                stop_on_rate_limit=stop_on_rate_limit,
            )
        except Exception:
            break
        if result is None:
            break
        await asyncio.sleep(page_sleep_s)

    return posts


def scrape_user_posts(
    username: str,
    max_posts: int,
    cookies_path: str,
    sleep_s: float = 1.0,
    retries: int = 4,
    *,
    page_sleep_s: float | None = None,
    page_size: int = 70,
    min_request_interval_s: float | None = None,
    max_rate_limit_hits: int = 5,
    stop_on_rate_limit: bool = False,
) -> list[dict[str, Any]]:
    if page_sleep_s is None:
        page_sleep_s = sleep_s
    if min_request_interval_s is None:
        min_request_interval_s = max(1.0, sleep_s)
    return asyncio.run(
        _scrape_async(
            username,
            max_posts,
            cookies_path,
            sleep_s,
            retries,
            page_sleep_s=page_sleep_s,
            page_size=page_size,
            min_request_interval_s=min_request_interval_s,
            max_rate_limit_hits=max_rate_limit_hits,
            stop_on_rate_limit=stop_on_rate_limit,
        )
    )
