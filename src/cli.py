from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.dedupe import merge_dedupe
from src.event_align import assign_event_day
from src.event_study import build_windows, compute_ar_car, fit_market_model
from src.io_raw import write_raw_dump
from src.market_data import compute_daily_returns, download_daily_prices
from src.report import summarize_car, write_report
from src.scrape_x import scrape_user_posts
from src.textproc import add_vader_sentiment, clean_text


def _ensure_dirs(*dirs: str) -> None:
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_scrape(config: dict) -> Path:
    scrape_cfg = config["scrape"]
    paths_cfg = config["paths"]

    posts = scrape_user_posts(
        scrape_cfg["username"],
        int(scrape_cfg["max_posts"]),
        scrape_cfg["cookies_path"],
        sleep_s=float(scrape_cfg.get("sleep_s", 1.0)),
        retries=int(scrape_cfg.get("retries", 4)),
    )

    raw_path = write_raw_dump(paths_cfg["raw_dir"], scrape_cfg["username"], posts)

    merged_path = Path(paths_cfg["processed_dir"]) / f"{scrape_cfg['username']}_merged.json"
    existing_posts = _load_json(merged_path)
    merged = merge_dedupe(existing_posts, posts)

    ids = [post.get("id") for post in merged if post.get("id")]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate event IDs found after merge/dedupe.")

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2)

    return raw_path


def run_process(config: dict) -> Path:
    scrape_cfg = config["scrape"]
    paths_cfg = config["paths"]

    merged_path = Path(paths_cfg["processed_dir"]) / f"{scrape_cfg['username']}_merged.json"
    posts = _load_json(merged_path)
    df = pd.DataFrame(posts)
    if df.empty:
        raise ValueError("No merged posts available to process.")

    df["content_raw"] = df["text"].fillna("")
    df["content_clean"] = df["content_raw"].apply(clean_text)
    df["date_utc"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = add_vader_sentiment(df)

    cleaned_path = Path(paths_cfg["processed_dir"]) / f"{scrape_cfg['username']}_cleaned.csv"
    df[
        [
            "id",
            "date_utc",
            "url",
            "user",
            "content_raw",
            "content_clean",
            "sent_compound",
            "sent_pos",
            "sent_neg",
            "sent_neu",
        ]
    ].to_csv(cleaned_path, index=False)

    return cleaned_path


def run_market(config: dict, posts_df: pd.DataFrame) -> tuple[Path, Path, pd.DataFrame]:
    paths_cfg = config["paths"]
    market_cfg = config["market"]

    symbols = list(dict.fromkeys([market_cfg["benchmark"], *market_cfg["symbols"]]))
    start = posts_df["date_utc"].min().date().isoformat()
    end = (posts_df["date_utc"].max() + pd.Timedelta(days=1)).date().isoformat()

    prices = download_daily_prices(symbols, start=start, end=end)
    if prices.empty:
        raise ValueError("No market data returned from yfinance.")

    returns = compute_daily_returns(prices, benchmark_symbol=market_cfg["benchmark"])

    returns = returns.copy()
    returns.index.name = "date"
    tidy = returns.reset_index().melt(id_vars="date", var_name="symbol", value_name="ret")
    returns_tidy_path = Path(paths_cfg["processed_dir"]) / "returns_tidy.csv"
    tidy.to_csv(returns_tidy_path, index=False)

    benchmark_returns = returns[[market_cfg["benchmark"]]].rename(
        columns={market_cfg["benchmark"]: "mkt_ret"}
    )
    benchmark_returns_path = Path(paths_cfg["processed_dir"]) / "benchmark_returns.csv"
    benchmark_returns.index.name = "date"
    benchmark_returns.reset_index().to_csv(benchmark_returns_path, index=False)

    return returns_tidy_path, benchmark_returns_path, returns


def run_align(config: dict, posts_df: pd.DataFrame, trading_days: pd.DatetimeIndex) -> Path:
    paths_cfg = config["paths"]
    event_cfg = config["event"]

    aligned = assign_event_day(
        posts_df,
        trading_days,
        event_cfg["market_tz"],
        event_cfg["open_time"],
        event_cfg["close_time"],
    )

    aligned_path = Path(paths_cfg["processed_dir"]) / "events_aligned_days.csv"
    aligned.to_csv(aligned_path, index=False)
    return aligned_path


def run_study(config: dict, aligned_df: pd.DataFrame, returns: pd.DataFrame) -> Path:
    paths_cfg = config["paths"]
    market_cfg = config["market"]
    event_cfg = config["event"]

    trading_days = returns.index
    event_window = tuple(event_cfg["event_window"])
    estimation_window = tuple(event_cfg["estimation_window"])

    event_results = []

    for _, row in aligned_df.iterrows():
        event_day = pd.Timestamp(row["event_day"])
        if event_day not in trading_days:
            raise ValueError("Event day not in trading calendar.")

        est_days, evt_days = build_windows(
            event_day,
            trading_days,
            est_win=estimation_window,
            evt_win=event_window,
        )

        if len(est_days) < 60:
            continue

        for symbol in market_cfg["symbols"]:
            if symbol == market_cfg["benchmark"]:
                continue
            ri_est = returns.loc[est_days, symbol]
            rm_est = returns.loc[est_days, market_cfg["benchmark"]]
            try:
                alpha, beta, resid_var = fit_market_model(ri_est, rm_est)
            except ValueError:
                continue

            ri_evt = returns.loc[evt_days, symbol]
            rm_evt = returns.loc[evt_days, market_cfg["benchmark"]]
            ar_series, car = compute_ar_car(alpha, beta, ri_evt, rm_evt)
            ar_values = ar_series.reindex(evt_days)

            event_results.append(
                {
                    "event_id": row["id"],
                    "event_day": event_day.date().isoformat(),
                    "symbol": symbol,
                    "car": car,
                    "ar_-1": ar_values.iloc[0] if len(ar_values) > 0 else None,
                    "ar_0": ar_values.iloc[1] if len(ar_values) > 1 else None,
                    "ar_+1": ar_values.iloc[2] if len(ar_values) > 2 else None,
                    "alpha": alpha,
                    "beta": beta,
                    "resid_var": resid_var,
                    "n_est": int(ri_est.dropna().shape[0]),
                }
            )

    results_df = pd.DataFrame(event_results)
    output_path = Path(paths_cfg["outputs_dir"]) / "event_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    return output_path


def run_report(config: dict, event_results_path: Path) -> Path:
    paths_cfg = config["paths"]
    df = pd.read_csv(event_results_path)
    summary = summarize_car(df)
    output_path = Path(paths_cfg["outputs_dir"]) / "report.json"
    write_report(str(output_path), summary)
    print(json.dumps(summary, indent=2))
    return output_path


def run_all(config_path: str) -> None:
    config = load_config(config_path)
    paths_cfg = config["paths"]
    _ensure_dirs(paths_cfg["raw_dir"], paths_cfg["processed_dir"], paths_cfg["outputs_dir"])

    run_scrape(config)

    cleaned_path = run_process(config)
    posts_df = pd.read_csv(cleaned_path, parse_dates=["date_utc"])

    _, _, returns = run_market(config, posts_df)

    aligned_path = run_align(config, posts_df, returns.index)
    aligned_df = pd.read_csv(aligned_path)

    event_results_path = run_study(config, aligned_df, returns)
    run_report(config, event_results_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Event study pipeline")
    parser.add_argument("--config", default="config.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run")
    subparsers.add_parser("scrape")
    subparsers.add_parser("process")
    subparsers.add_parser("market")
    subparsers.add_parser("study")
    subparsers.add_parser("report")

    args = parser.parse_args()
    config = load_config(args.config)
    paths_cfg = config["paths"]
    _ensure_dirs(paths_cfg["raw_dir"], paths_cfg["processed_dir"], paths_cfg["outputs_dir"])

    if args.command == "run":
        run_all(args.config)
        return

    if args.command == "scrape":
        run_scrape(config)
        return

    if args.command == "process":
        run_process(config)
        return

    if args.command == "market":
        cleaned_path = Path(paths_cfg["processed_dir"]) / f"{config['scrape']['username']}_cleaned.csv"
        posts_df = pd.read_csv(cleaned_path, parse_dates=["date_utc"])
        run_market(config, posts_df)
        return

    if args.command == "study":
        aligned_path = Path(paths_cfg["processed_dir"]) / "events_aligned_days.csv"
        aligned_df = pd.read_csv(aligned_path)
        returns = pd.read_csv(
            Path(paths_cfg["processed_dir"]) / "returns_tidy.csv",
            parse_dates=["date"],
        )
        returns_wide = returns.pivot(index="date", columns="symbol", values="ret")
        run_study(config, aligned_df, returns_wide)
        return

    if args.command == "report":
        event_results_path = Path(paths_cfg["outputs_dir"]) / "event_results.csv"
        run_report(config, event_results_path)
        return


if __name__ == "__main__":
    main()
