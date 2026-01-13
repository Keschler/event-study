from __future__ import annotations

from datetime import time

import pandas as pd
import pytz


def _parse_time(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def assign_event_day(
    posts_df: pd.DataFrame,
    trading_days_index: pd.DatetimeIndex,
    market_tz: str,
    open_time: str,
    close_time: str,
) -> pd.DataFrame:
    tz = pytz.timezone(market_tz)
    open_t = _parse_time(open_time)
    close_t = _parse_time(close_time)

    trading_days_idx = pd.DatetimeIndex(trading_days_index)
    if trading_days_idx.tz is None:
        trading_days_idx = trading_days_idx.tz_localize(tz, nonexistent="NaT", ambiguous="NaT")
    else:
        trading_days_idx = trading_days_idx.tz_convert(tz)
    trading_days = trading_days_idx.date
    trading_days_set = set(trading_days)

    def next_trading_day(day: pd.Timestamp) -> pd.Timestamp:
        date = day.date()
        while date not in trading_days_set:
            date = (pd.Timestamp(date) + pd.Timedelta(days=1)).date()
        return pd.Timestamp(date).tz_localize(tz)

    def assign(ts_utc: pd.Timestamp) -> pd.Timestamp:
        ts_local = ts_utc.tz_convert(tz)
        day = ts_local.normalize()
        if ts_local.date() not in trading_days_set:
            return next_trading_day(ts_local)
        if open_t <= ts_local.time() <= close_t:
            return day
        if ts_local.time() > close_t:
            return next_trading_day(ts_local + pd.Timedelta(days=1))
        return day

    output = posts_df.copy()
    output["date_utc"] = pd.to_datetime(output["created_at"], utc=True, errors="coerce")
    output["event_day"] = output["date_utc"].apply(assign)
    output["event_day"] = output["event_day"].dt.date
    return output
