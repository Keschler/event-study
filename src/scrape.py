import asyncio
import json
from twikit import Client

client = Client('en-US')  # make sure you've already called client.load_cookies("cookies.json")
client.load_cookies("cookies.json")

async def run():
    username = "trump_repost"
    max_tweets = 500

    # Resolve user ID from screen name
    user = await client.get_user_by_screen_name(username)  # this is sync in current twikit
    user_id = user.id

    out = []

    # First page
    result = await client.get_user_tweets(user_id, 'Tweets', count=min(100, max_tweets))

    def to_dict(tw):
        # Build a URL: https://x.com/<screen_name>/status/<id>
        url = f"https://x.com/{tw.user.screen_name}/status/{tw.id}"
        # created_at_datetime is a Python datetime; fall back to created_at (string) if needed
        dt = getattr(tw, "created_at_datetime", None)
        iso = dt.isoformat() if dt else getattr(tw, "created_at", "")
        return {
            "id": tw.id,
            "date": iso,
            "content": tw.full_text,
            "url": url,
            "user": tw.user.screen_name,
        }

    # Collect with pagination
    while True:
        for tw in result:
            if hasattr(tw, "retweeted_tweet") or hasattr(tw, "retweeted_status"):
                continue
            out.append(to_dict(tw))
            if len(out) >= max_tweets:
                break
        if len(out) >= max_tweets:
            break
        # Fetch next page if available
        try:
            result = await result.next()  # Result has .next() for pagination
        except Exception:
            break  # no more pages

    # Save
    with open(f"{username}_tweets.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(out)} tweets to {username}_tweets.json")


if __name__ == "__main__":
    asyncio.run(run())

