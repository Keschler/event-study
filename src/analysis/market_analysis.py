import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import json
import pytz
import os
from datetime import timezone
from pathlib import Path
from nltk.sentiment.vader import SentimentIntensityAnalyzer



UTC = timezone.utc
MARKET_TZ = pytz.timezone("US/Eastern")

EVENT_WINDOW_HOURS = 2
MARKET_OPEN_H = 9
MARKET_OPEN_M = 30

MARKET_CLOSE_H = 16
MARKET_CLOSE_M = 0
TWEETS_JSON = "trump_repost_tweets.json"

BAR_INTERVAL = "1h"
INDEX_SYMBOL = None


config_path = Path(__file__).resolve().parents[2] / "config.json"
if config_path.is_file():
    with config_path.open("r", encoding="utf-8") as f:
        config_data = json.load(f)
    INDEX_SYMBOL = config_data.get("market", {}).get("benchmark")
    BAR_INTERVAL = config_data.get("market", {}).get("bar_interval", BAR_INTERVAL)

if INDEX_SYMBOL is None:
    raise ValueError("INDEX_SYMBOL must be set via config.json market.benchmark.")

print(INDEX_SYMBOL, BAR_INTERVAL)

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

def build_interpretation_text(events, reg_df, model, summary, index_name="Nasdaq Composite"):
    lines = []

    n_events = summary["n_events"]
    n_reg = summary["n_regular_hours"]
    mean_impact = summary["mean_post_minus_pre"]          # log return
    mean_impact_pct = mean_impact * 100                   # approx %
    mean_vol_change = summary["mean_vol_post_minus_pre"]
    sent_mean = summary["sentiment_compound_mean"]
    sent_std = summary["sentiment_compound_std"]

    lines.append(f"Event study interpretation for {index_name}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Number of tweets (events): {n_events}")
    lines.append(f"Tweets during regular US trading hours: {n_reg}")
    lines.append("")
    lines.append("Average price impact (log returns):")
    lines.append(f"  mean(post - pre): {mean_impact:.6f} ≈ {mean_impact_pct:.4f}%")
    lines.append(f"Average change in volatility (post - pre): {mean_vol_change:.6f}")
    lines.append("")
    lines.append("Tweet sentiment (VADER compound):")
    lines.append(f"  mean: {sent_mean:.4f}, std: {sent_std:.4f}")
    lines.append("")

    if model is None or len(reg_df) < 5:
        lines.append("Regression: not enough valid events to estimate a reliable relationship.")
        return "\n".join(lines)

    # Regression details
    params = model.params
    pvalues = model.pvalues
    rsq = model.rsquared

    # We expect X = [const, sent_compound]
    alpha = float(params[0])
    beta = float(params[1]) if len(params) > 1 else float("nan")
    p_beta = float(pvalues[1]) if len(pvalues) > 1 else float("nan")

    beta_pct = beta * 100  # percentage impact per 1.0 change in sentiment

    lines.append("Regression: return impact vs. tweet sentiment")
    lines.append("----------------------------------------------")
    lines.append("Model: ret_post_minus_pre = alpha + beta * sentiment_compound + error")
    lines.append(f"alpha: {alpha:.6e}")
    lines.append(f"beta:  {beta:.6e}  (≈ {beta_pct:.6f}% per 1.0 sentiment unit)")
    lines.append(f"p-value(beta): {p_beta:.4g}")
    lines.append(f"R²: {rsq:.4f}")
    lines.append("")

    # Qualitative interpretation of beta
    if not np.isfinite(beta):
        lines.append("Could not interpret beta because it is NaN or infinite.")
        return "\n".join(lines)

    if p_beta < 0.01:
        sig_text = "highly statistically significant (p < 0.01)"
    elif p_beta < 0.05:
        sig_text = "statistically significant at the 5% level (p < 0.05)"
    elif p_beta < 0.1:
        sig_text = "weakly statistically significant (p < 0.10)"
    else:
        sig_text = "not statistically significant (p ≥ 0.10)"

    if beta > 0:
        dir_text = "On average, more positive tweet sentiment is associated with higher post-minus-pre returns."
    elif beta < 0:
        dir_text = "On average, more positive tweet sentiment is associated with lower post-minus-pre returns."
    else:
        dir_text = "The estimated effect of sentiment on returns is essentially zero."

    lines.append("Interpretation of regression coefficient:")
    lines.append(f"- The estimated beta is {beta:.6e}, which is {sig_text}.")
    lines.append(f"- {dir_text}")
    lines.append(f"- A full change of sentiment from -1.0 to +1.0 would correspond to about {2*beta_pct:.6f}% change")
    lines.append("  in the post-minus-pre log return, if the linear model is taken literally.")
    lines.append("")
    lines.append("Note:")
    lines.append("- R² indicates how much of the variation in return impact is explained by sentiment alone.")
    lines.append("- Even a statistically significant beta with low R² means sentiment moves the index a bit,")
    lines.append("  but most of the movement is still driven by other factors (news, macro data, noise, etc.).")
    lines.append("Bottom line:")
    if abs(mean_impact_pct) < 0.01:
        lines.append("- On average, the price impact around these tweets is extremely small in economic terms.")
    elif abs(mean_impact_pct) < 0.1:
        lines.append("- The average impact is modest but potentially noticeable on an intraday basis.")
    else:
        lines.append("- The average impact is large enough to be economically meaningful.")

    if p_beta < 0.05:
        lines.append("- There is statistically significant evidence that sentiment matters for short-term moves.")
    else:
        lines.append("- The statistical evidence that sentiment matters is weak in this sample.")


    return "\n".join(lines)



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

def get_vader_analyzer() -> SentimentIntensityAnalyzer:
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        import nltk

        nltk.download("vader_lexicon")
        return SentimentIntensityAnalyzer()


tweets["cleaned"] = tweets["content"].apply(clean_text)

# VADER sentiment
an = get_vader_analyzer()
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
# Does more positive tweet sentiment correspond to a larger positive return after the tweet relative to before
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

# Build interpretation text (pass model or None depending on whether regression ran)
interp_model = model if ("model" in locals() and len(reg_df) >= 5) else None
interpretation = build_interpretation_text(events, reg_df if "reg_df" in locals() else events, interp_model, summary, index_name="Nasdaq Composite")

# Print to console
print("\n\n=== HUMAN-READABLE INTERPRETATION ===\n")
print(interpretation)

# Save to file
with open("Results/interpretation.txt", "w") as f:
    f.write(interpretation)
print("Saved interpretation -> Results/interpretation.txt")



# Scatter: sentiment vs. return impact
plt.figure()
plt.scatter(reg_df.get("sent_compound", []), reg_df.get("ret_post_minus_pre", []), alpha=0.6)
plt.xlabel("Tweet sentiment (compound)")
plt.ylabel("Return impact (post - pre) [log return]")
plt.title("Nasdaq: sentiment vs. return impact")
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
