"""Twitter/X scraping module with retry logic and rate limiting."""

import asyncio
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from twikit import Client


async def scrape_user_posts(
    username: str,
    max_posts: int,
    cookies_path: str,
    sleep_s: float = 1.0,
    retries: int = 3
) -> List[Dict[str, Any]]:
    """
    Scrape posts from a Twitter/X user with retry logic and rate limiting.
    
    Args:
        username: Twitter/X username to scrape
        max_posts: Maximum number of posts to retrieve
        cookies_path: Path to cookies.json file
        sleep_s: Seconds to sleep between requests (rate limiting)
        retries: Number of retries on failure
        
    Returns:
        List of post dictionaries with keys: id, date, content, url, user
        
    Raises:
        Exception: If scraping fails after all retries
    """
    client = Client('en-US')
    
    cookies_file = Path(cookies_path)
    if not cookies_file.exists():
        raise FileNotFoundError(f"Cookies file not found: {cookies_path}")
    
    client.load_cookies(str(cookies_file))
    
    posts = []
    attempt = 0
    
    while attempt < retries:
        try:
            # Resolve user ID from screen name
            user = await client.get_user_by_screen_name(username)
            user_id = user.id
            
            # First page
            result = await client.get_user_tweets(
                user_id, 'Tweets', count=min(100, max_posts)
            )
            
            # Collect with pagination
            while True:
                for tw in result:
                    # Skip retweets/reposts
                    if hasattr(tw, "retweeted_tweet") or hasattr(tw, "retweeted_status"):
                        continue
                    
                    # Build post dictionary
                    url = f"https://x.com/{tw.user.screen_name}/status/{tw.id}"
                    dt = getattr(tw, "created_at_datetime", None)
                    iso = dt.isoformat() if dt else getattr(tw, "created_at", "")
                    
                    posts.append({
                        "id": tw.id,
                        "date": iso,
                        "content": tw.full_text,
                        "url": url,
                        "user": tw.user.screen_name,
                    })
                    
                    if len(posts) >= max_posts:
                        break
                
                if len(posts) >= max_posts:
                    break
                
                # Rate limiting between page requests
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                
                # Fetch next page if available
                try:
                    result = await result.next()
                except Exception:
                    # No more pages
                    break
            
            # Success - return posts
            return posts
            
        except Exception as e:
            attempt += 1
            if attempt >= retries:
                raise Exception(
                    f"Failed to scrape {username} after {retries} attempts: {e}"
                )
            
            # Exponential backoff: 1s, 2s, 4s, ...
            backoff = min(2 ** (attempt - 1), 8)
            print(f"Scraping attempt {attempt} failed: {e}")
            print(f"Retrying in {backoff} seconds...")
            await asyncio.sleep(backoff)
    
    return posts


async def scrape_user_posts_cli(
    username: str,
    max_posts: int,
    cookies_path: str,
    output_path: Optional[str] = None,
    sleep_s: float = 1.0,
    retries: int = 3
) -> str:
    """
    CLI wrapper for scraping that saves results to a file.
    
    Args:
        username: Twitter/X username to scrape
        max_posts: Maximum number of posts to retrieve
        cookies_path: Path to cookies.json file
        output_path: Path to save results (default: {username}_tweets.json)
        sleep_s: Seconds to sleep between requests
        retries: Number of retries on failure
        
    Returns:
        Path to the saved file
    """
    posts = await scrape_user_posts(
        username, max_posts, cookies_path, sleep_s, retries
    )
    
    if output_path is None:
        output_path = f"{username}_tweets.json"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(posts)} posts to {output_path}")
    return output_path
