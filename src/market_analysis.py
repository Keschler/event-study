import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import json
import nltk
import pytz
from datetime import timezone
from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer




UTC = timezone.utc

nltk.download('vader_lexicon')
MARKET_TZ = pytz.timezone("US/Eastern")
INDEX_SYMBOL = "^IXIC"

EVENT_WINDOW_HOURS = 2
BAR_INTERVAL = "1h"
MARKET_OPEN_H = 9
MARKET_OPEN_M = 30

MARKET_CLOSE_H = 16
MARKET_CLOSE_M = 0
TWEETS_JSON = "trump_repost_tweets.json"


# ----------------------------
# Helpers
# ----------------------------
def within_regular_hours(ts_et: pd.Timestamp) -> bool:
    # Regular session 09:30–16:00 ET (no holiday check)
    op = ts_et.replace(hour=MARKET_OPEN_H, minute=MARKET_OPEN_M, second=0, microsecond=0)
    cl = ts_et.replace(hour=MARKET_CLOSE_H, minute=MARKET_CLOSE_M, second=0, microsecond=0)
    return (ts_et >= op) and (ts_et <= cl)


def nearest_index_time(idx_series: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Timestamp:
    # Return index timestamp closest to t
    pos = idx_series.get_indexer([t], method="nearest")[0]
    return idx_series[pos]

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def compute_window_returns(px: pd.Series, t: pd.Timestamp, h: int) -> dict:
    """Compute log-returns in [-h, +h] around time t on a pandas Series of prices."""
    px = px.sort_index()
    start = t - pd.Timedelta(hours=h)
    end = t + pd.Timedelta(hours=h)

    w = px.loc[start:end]
    pre = px.loc[start:t]
    post = px.loc[t:end]

    out = {}

    # total window
    if len(w) >= 2:
        out["ret_total_log"] = _to_float(np.log(_to_float(w.iloc[-1]) / _to_float(w.iloc[0])))
    else:
        out["ret_total_log"] = np.nan

    # pre window
    if len(pre) >= 2:
        out["ret_pre_log"]  = _to_float(np.log(_to_float(pre.iloc[-1]) / _to_float(pre.iloc[0])))
    else:
        out["ret_pre_log"] = np.nan

    # post window
    if len(post) >= 2:
        out["ret_post_log"] = _to_float(np.log(_to_float(post.iloc[-1]) / _to_float(post.iloc[0])))
    else:
        out["ret_post_log"] = np.nan

    # difference
    pre_ok  = np.isfinite(out["ret_pre_log"])
    post_ok = np.isfinite(out["ret_post_log"])
    out["ret_post_minus_pre"] = _to_float(out["ret_post_log"] - out["ret_pre_log"]) if (pre_ok and post_ok) else np.nan

    return out


def compute_window_vol(px: pd.Series, t: pd.Timestamp, h: int) -> dict:
    """Std of hourly log-returns pre/post as a realized volatility proxy."""
    px = px.sort_index()
    ret = np.log(px).diff().dropna()

    pre = ret.loc[t - pd.Timedelta(hours=h): t]
    post = ret.loc[t: t + pd.Timedelta(hours=h)]

    vol_pre = _to_float(pre.std()) if len(pre) > 1 else np.nan
    vol_post = _to_float(post.std()) if len(post) > 1 else np.nan

    out = {
        "vol_pre": vol_pre,
        "vol_post": vol_post,
        "vol_post_minus_pre": _to_float(vol_post - vol_pre) if (np.isfinite(vol_pre) and np.isfinite(vol_post)) else np.nan,
    }
    return out

# ----------------------------
# 1) Load tweets and sentiment
# ----------------------------
tweets_path = Path(TWEETS_JSON)
if not tweets_path.exists():
    raise FileNotFoundError(f"Place {TWEETS_JSON} next to this script.")

with open(tweets_path, "r", encoding="utf-8") as f:
    tweets_json = json.load(f)

tweets = pd.DataFrame(tweets_json)
if "date" not in tweets.columns or "content" not in tweets.columns:
    raise ValueError("Expected keys 'date' and 'content' in the tweet JSON.")

# Timestamps -> UTC
tweets["date"] = pd.to_datetime(tweets["date"], utc=True, errors="coerce")
tweets = tweets.dropna(subset=["date"]).sort_values("date").reset_index(drop=True) # Remove NULL values

# Basic text cleanup
def clean_text(text: str) -> str:
    import re
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s']", " ", text)
    return " ".join(text.split())

tweets["cleaned"] = tweets["content"].apply(clean_text)

# VADER sentiment
an = SentimentIntensityAnalyzer()
scores = tweets["cleaned"].apply(lambda s: an.polarity_scores(s))
tweets["sent_pos"] = scores.apply(lambda d: d["pos"])
tweets["sent_neg"] = scores.apply(lambda d: d["neg"])
tweets["sent_neu"] = scores.apply(lambda d: d["neu"])
tweets["sent_compound"] = scores.apply(lambda d: d["compound"])

# Keep only tweets within or near market hours for intraday alignment
tweets["date_et"] = tweets["date"].dt.tz_convert(MARKET_TZ)
tweets["is_regular_hours"] = tweets["date_et"].apply(within_regular_hours)
# You can choose to keep all tweets; here we keep both but mark the flag
print(f"Loaded {len(tweets)} tweets; {tweets['is_regular_hours'].sum()} during regular market hours.")


# ----------------------------
# 2) Download S&P 500 hourly
# ----------------------------
tmin = tweets["date"].min() - pd.Timedelta(days=7)
tmax = tweets["date"].max() + pd.Timedelta(days=7)

print("Downloading market data...")
px = yf.download(INDEX_SYMBOL, start=tmin.date().isoformat(), end=(tmax + pd.Timedelta(days=1)).date().isoformat(),
                 interval=BAR_INTERVAL, auto_adjust=False, progress=False)

if px.empty:
    raise RuntimeError("No market data returned by yfinance. Try a wider date range or check connectivity.")

# Use Close; ensure UTC
px.index = px.index.tz_localize(None).tz_localize(UTC)
close = px["Close"].dropna().sort_index()

# For reference, hourly log returns
logret = np.log(close).diff()


# ----------------------------
# 3) Align tweets to nearest bar and compute event metrics
# ----------------------------
aligned_records = []

for i, row in tweets.iterrows():
    t_utc = row["date"]
    # Snap to nearest available hourly bar
    t_bar = nearest_index_time(close.index, t_utc)

    # Compute returns around t_bar
    ret_stats = compute_window_returns(close, t_bar, EVENT_WINDOW_HOURS)
    vol_stats = compute_window_vol(close, t_bar, EVENT_WINDOW_HOURS)

    aligned_records.append({
        "tweet_id": row.get("id", ""),
        "tweet_time_utc": t_utc,
        "bar_time_utc": t_bar,
        "user": row.get("user", ""),
        "url": row.get("url", ""),
        "content": row.get("content", ""),
        "cleaned": row.get("cleaned", ""),
        "is_regular_hours": row["is_regular_hours"],
        "sent_pos": row["sent_pos"],
        "sent_neg": row["sent_neg"],
        "sent_neu": row["sent_neu"],
        "sent_compound": row["sent_compound"],
        **ret_stats,
        **vol_stats
    })

events = pd.DataFrame(aligned_records).sort_values("bar_time_utc").reset_index(drop=True)
events.to_csv("events_aligned.csv", index=False)
print(f"Saved per-tweet event metrics -> events_aligned.csv  (rows: {len(events)})")


# ----------------------------
# 4) Simple regression: impact vs sentiment
# ----------------------------
# We'll use post-minus-pre log return as 'impact'
reg_df = events.dropna(subset=["ret_post_minus_pre", "sent_compound"]).copy()
if len(reg_df) >= 5:
    X = sm.add_constant(reg_df["sent_compound"].values)
    y = reg_df["ret_post_minus_pre"].values
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Save regression summary
    with open("regression_summary.txt", "w") as f:
        f.write(model.summary().as_text())
    print("Saved OLS summary -> regression_summary.txt")
else:
    print("Not enough observations with non-NaN to run regression.")


# ----------------------------
# 5) Aggregate summaries and charts
# ----------------------------
summary = {
    "n_events": int(len(events)),
    "n_regular_hours": int(events["is_regular_hours"].sum()),
    "mean_total_log_return": float(events["ret_total_log"].mean(skipna=True)),
    "mean_post_minus_pre": float(events["ret_post_minus_pre"].mean(skipna=True)),
    "mean_vol_post_minus_pre": float(events["vol_post_minus_pre"].mean(skipna=True)),
    "sentiment_compound_mean": float(events["sent_compound"].mean(skipna=True)),
    "sentiment_compound_std": float(events["sent_compound"].std(skipna=True)),
}
pd.DataFrame([summary]).to_csv("Results/event_summary.csv", index=False)
print("Saved overall summary -> Results/event_summary.csv")

# Scatter: sentiment vs. return impact
plt.figure()
plt.scatter(reg_df.get("sent_compound", []), reg_df.get("ret_post_minus_pre", []), alpha=0.6)
plt.xlabel("Tweet sentiment (compound)")
plt.ylabel("Return impact (post - pre) [log return]")
plt.title("S&P 500 hourly: sentiment vs. return impact")
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig("Results/scatter_sentiment_vs_impact.png", dpi=160)
plt.close()
print("Saved chart -> Results/scatter_sentiment_vs_impact.png")

# Histogram of impacts
plt.figure()
imp = events["ret_post_minus_pre"].dropna()
plt.hist(imp, bins=40)
plt.xlabel("Return impact (post - pre) [log return]")
plt.ylabel("Count")
plt.title("Distribution of event impacts")
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig("Results/hist_event_impacts.png", dpi=160)
plt.close()
print("Saved chart -> Results/hist_event_impacts.png")

print("Done.")


