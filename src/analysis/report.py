from __future__ import annotations

import json
from math import sqrt

import numpy as np
import pandas as pd


def summarize_car(event_results_df: pd.DataFrame) -> dict:
    car_series = event_results_df["car"].dropna()
    n = int(car_series.shape[0])
    mean_car = float(car_series.mean()) if n else float("nan")
    std_car = float(car_series.std(ddof=1)) if n > 1 else float("nan")

    t_stat = float("nan")
    p_value = float("nan")

    if n > 1 and np.isfinite(std_car) and std_car > 0:
        try:
            from scipy.stats import ttest_1samp

            t_res = ttest_1samp(car_series, 0.0, nan_policy="omit")
            t_stat = float(t_res.statistic)
            p_value = float(t_res.pvalue)
        except Exception:
            t_stat = mean_car / (std_car / sqrt(n))
            p_value = float("nan")

    return {
        "n": n,
        "mean_car": mean_car,
        "std_car": std_car,
        "t_stat": t_stat,
        "p_value": p_value,
    }


def write_report(path: str, summary: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
