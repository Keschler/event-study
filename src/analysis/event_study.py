from __future__ import annotations

import numpy as np
import pandas as pd


def build_windows(
    event_day: pd.Timestamp,
    trading_days: pd.DatetimeIndex,
    est_win: tuple[int, int] = (-120, -21),
    evt_win: tuple[int, int] = (-1, 1),
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    trading_days = pd.DatetimeIndex(trading_days)
    if event_day not in trading_days:
        raise ValueError("Event day must be in trading_days index.")

    idx = trading_days.get_loc(event_day)
    est_start = idx + est_win[0]
    est_end = idx + est_win[1]
    evt_start = idx + evt_win[0]
    evt_end = idx + evt_win[1]

    est_days = trading_days[est_start : est_end + 1]
    evt_days = trading_days[evt_start : evt_end + 1]
    return est_days, evt_days


def fit_market_model(
    ri_est: pd.Series,
    rm_est: pd.Series,
) -> tuple[float, float, float]:
    aligned = pd.concat([ri_est, rm_est], axis=1).dropna()
    if len(aligned) < 3:
        raise ValueError("Not enough data to fit market model.")

    y = aligned.iloc[:, 0].values
    x = aligned.iloc[:, 1].values
    x_mat = np.column_stack([np.ones(len(x)), x])
    try:
        beta_hat, residuals, rank, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Regression fit failed.") from exc

    if rank < 2:
        raise ValueError("Regression matrix is singular.")

    alpha, beta = beta_hat
    resid = y - (alpha + beta * x)
    resid_var = float(np.var(resid, ddof=2))
    return float(alpha), float(beta), resid_var


def compute_ar_car(
    alpha: float,
    beta: float,
    ri_evt: pd.Series,
    rm_evt: pd.Series,
) -> tuple[pd.Series, float]:
    aligned = pd.concat([ri_evt, rm_evt], axis=1)
    expected = alpha + beta * aligned.iloc[:, 1]
    ar = aligned.iloc[:, 0] - expected
    car = float(ar.sum(skipna=True))
    return ar, car
