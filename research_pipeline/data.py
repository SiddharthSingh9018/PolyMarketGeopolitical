from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import requests
import yfinance as yf

from research_pipeline.config import PipelineConfig
from update_utils.update_markets import update_markets

POLY_ARCHIVE_URL = "https://polydata-archive.s3.us-east-1.amazonaws.com/archive.tar.xz"
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"


def prepare_directories(config: PipelineConfig) -> None:
    for directory in [
        config.raw_dir,
        config.interim_dir,
        config.processed_dir,
        config.plot_dir,
        config.table_dir,
        config.text_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    return destination


def ensure_polymarket_inputs(config: PipelineConfig) -> dict[str, str]:
    metadata: dict[str, str] = {"bootstrap_source": "local"}
    if config.markets_path.exists() and config.trades_path.exists():
        return metadata

    if not config.markets_path.exists():
        update_markets(csv_filename=str(config.markets_path))
        metadata["markets_source"] = "gamma_api"

    if config.trades_path.exists():
        return metadata

    archive_path = config.raw_dir / "archive.tar.xz"
    if not archive_path.exists():
        metadata["bootstrap_source"] = "s3_archive_download"
        _download_file(POLY_ARCHIVE_URL, archive_path)

    with tarfile.open(archive_path, mode="r:xz") as archive:
        wanted_members = [
            member
            for member in archive.getmembers()
            if member.name.endswith("markets.csv") or member.name.endswith("processed/trades.csv")
        ]
        archive.extractall(path=config.root_dir, members=wanted_members)

    if not config.trades_path.exists():
        raise FileNotFoundError(
            "Polymarket trades were not found after archive extraction. "
            "Inspect the downloaded archive structure before rerunning."
        )
    return metadata


def load_markets(config: PipelineConfig) -> pd.DataFrame:
    markets = pd.read_csv(config.markets_path)
    markets["question"] = markets["question"].fillna("")
    markets["answer1"] = markets["answer1"].fillna("")
    markets["answer2"] = markets["answer2"].fillna("")
    markets["volume"] = pd.to_numeric(markets["volume"], errors="coerce")
    return markets


def select_relevant_markets(markets: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    if config.market_override_path and config.market_override_path.exists():
        override = pd.read_csv(config.market_override_path)
        selected = markets.merge(override[["id"]].drop_duplicates(), on="id", how="inner")
        if not selected.empty:
            return selected

    regex = "|".join(config.keyword_patterns)
    is_relevant = markets["question"].str.contains(regex, case=False, regex=True)
    is_yes_no = (
        markets["answer1"].str.lower().eq("yes") & markets["answer2"].str.lower().eq("no")
    )
    selected = markets.loc[is_relevant & is_yes_no].copy()
    selected = selected.loc[selected["volume"].fillna(0.0) >= config.min_market_volume].copy()
    return selected.sort_values("volume", ascending=False)


def build_polymarket_panel(config: PipelineConfig, selected_markets: pd.DataFrame) -> pd.DataFrame:
    selected_ids = selected_markets["id"].astype(str).drop_duplicates().tolist()
    lazy_trades = (
        pl.scan_csv(
            config.trades_path,
            schema_overrides={
                "timestamp": pl.Utf8,
                "market_id": pl.Utf8,
                "nonusdc_side": pl.Utf8,
                "taker_direction": pl.Utf8,
                "price": pl.Float64,
                "usd_amount": pl.Float64,
            },
        )
        .filter(pl.col("market_id").is_in(selected_ids))
        .filter(pl.col("nonusdc_side") == "token1")
        .with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S%.f", strict=False),
            pl.col("usd_amount").fill_null(0.0),
            pl.when(pl.col("taker_direction") == "BUY")
            .then(pl.col("usd_amount"))
            .otherwise(0.0)
            .alias("buy_volume"),
            pl.when(pl.col("taker_direction") == "SELL")
            .then(pl.col("usd_amount"))
            .otherwise(0.0)
            .alias("sell_volume"),
        )
        .drop_nulls(["timestamp", "price"])
        .with_columns(pl.col("timestamp").dt.truncate("1d").alias("date"))
        .sort(["market_id", "timestamp"])
    )

    market_daily = (
        lazy_trades.group_by(["market_id", "date"])
        .agg(
            pl.col("price").last().alias("probability_level"),
            pl.col("usd_amount").sum().alias("daily_volume"),
            pl.len().alias("daily_trades"),
            pl.col("buy_volume").sum().alias("buy_volume"),
            pl.col("sell_volume").sum().alias("sell_volume"),
        )
        .with_columns(
            pl.when((pl.col("buy_volume") + pl.col("sell_volume")) > 0)
            .then((pl.col("buy_volume") - pl.col("sell_volume")) / (pl.col("buy_volume") + pl.col("sell_volume")))
            .otherwise(0.0)
            .alias("order_imbalance"),
            pl.when(pl.col("daily_volume") > 0).then(pl.col("daily_volume")).otherwise(1.0).alias("weight"),
        )
    )

    agg = (
        market_daily.group_by("date")
        .agg(
            ((pl.col("probability_level") * pl.col("weight")).sum() / pl.col("weight").sum()).alias(
                "poly_probability_level"
            ),
            pl.col("daily_volume").sum().alias("poly_daily_volume"),
            pl.col("daily_trades").sum().alias("poly_trade_count"),
            ((pl.col("order_imbalance") * pl.col("weight")).sum() / pl.col("weight").sum()).alias(
                "poly_order_imbalance"
            ),
            pl.col("market_id").n_unique().alias("poly_market_count"),
        )
        .sort("date")
        .collect()
    )
    agg = pd.DataFrame(agg.to_dicts())

    agg["poly_probability_change"] = agg["poly_probability_level"].diff()
    agg["poly_probability_volatility"] = (
        agg["poly_probability_level"].rolling(config.probability_window).std()
    )
    rolling_volume = agg["poly_daily_volume"].rolling(config.volume_window)
    agg["poly_volume_zscore"] = (
        agg["poly_daily_volume"] - rolling_volume.mean()
    ) / rolling_volume.std(ddof=0)
    agg["regime"] = pd.cut(
        agg["poly_probability_level"],
        bins=[-np.inf, 0.3, 0.6, np.inf],
        labels=["low", "medium", "high"],
    )
    return agg


def download_market_series(config: PipelineConfig, tickers: list[str]) -> pd.DataFrame:
    history = yf.download(
        tickers=tickers,
        start="2019-01-01",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        multi_level_index=True,
        threads=False,
    )
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        ticker_frame = history[ticker].reset_index().rename(columns=str.lower)
        ticker_frame["asset"] = ticker
        frames.append(ticker_frame[["date", "asset", "open", "high", "low", "close", "volume"]])
    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.floor(config.frequency)
    return prices.sort_values(["asset", "date"])


def download_macro_series(config: PipelineConfig) -> pd.DataFrame:
    tickers = list(config.benchmark_tickers.values())
    raw = yf.download(
        tickers=tickers,
        start="2019-01-01",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        multi_level_index=True,
        threads=False,
    )
    series_map: dict[str, pd.Series] = {}
    for name, ticker in config.benchmark_tickers.items():
        ticker_frame = raw[ticker].reset_index().rename(columns=str.lower)
        series_map[name] = ticker_frame.set_index("date")["close"]
    macro = pd.concat(series_map, axis=1).reset_index()
    macro["date"] = pd.to_datetime(macro["date"]).dt.floor(config.frequency)
    macro = macro.rename(columns={"ovx": "oil_volatility_proxy", "wti": "wti_price"})
    return macro.sort_values("date")


def download_gpr_series(config: PipelineConfig) -> pd.DataFrame:
    cached_path = config.raw_dir / "data_gpr_daily_recent.xls"
    if not cached_path.exists():
        _download_file(GPR_URL, cached_path)

    workbook = pd.read_excel(cached_path, sheet_name=0)
    date_col = next(col for col in workbook.columns if "date" in str(col).lower())
    gpr_col = next(col for col in workbook.columns if "gpr" in str(col).lower())
    gpr = workbook[[date_col, gpr_col]].copy()
    gpr.columns = ["date", "gpr"]
    gpr["date"] = pd.to_datetime(gpr["date"]).dt.floor(config.frequency)
    gpr["gpr"] = pd.to_numeric(gpr["gpr"], errors="coerce")
    return gpr.dropna(subset=["date", "gpr"]).sort_values("date")


def load_sentiment_series(
    config: PipelineConfig, prices: pd.DataFrame, macro: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    if config.sentiment_path and config.sentiment_path.exists():
        sentiment = pd.read_csv(config.sentiment_path, parse_dates=["date"])
        sentiment["date"] = sentiment["date"].dt.floor(config.frequency)
        sentiment["sentiment"] = pd.to_numeric(sentiment["sentiment"], errors="coerce")
        return sentiment[["date", "sentiment"]].dropna(), "external"

    basket = (
        prices.groupby("date", as_index=False)
        .agg(defense_return_proxy=("close", "mean"))
        .sort_values("date")
    )
    basket["defense_return_proxy"] = np.log(
        basket["defense_return_proxy"] / basket["defense_return_proxy"].shift(1)
    )
    sentiment = basket.merge(macro[["date", "vix"]], on="date", how="left")
    sentiment["sentiment"] = -0.7 * sentiment["defense_return_proxy"].fillna(0.0) - 0.3 * (
        sentiment["vix"].pct_change().fillna(0.0)
    )
    return sentiment[["date", "sentiment"]], "proxy"


def save_market_selection(selected_markets: pd.DataFrame, config: PipelineConfig) -> None:
    path = config.table_dir / "selected_polymarket_markets.csv"
    selected_markets.to_csv(path, index=False)


def save_data_manifest(
    config: PipelineConfig,
    bootstrap_meta: dict[str, str],
    sentiment_source: str,
    selected_markets: pd.DataFrame,
) -> None:
    manifest = {
        "frequency": config.frequency,
        "forecast_horizon": config.forecast_horizon,
        "realized_window": config.realized_window,
        "selected_market_count": int(selected_markets["id"].nunique()),
        "asset_tickers": config.asset_tickers,
        "sentiment_source": sentiment_source,
        **bootstrap_meta,
    }
    manifest_path = config.text_dir / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
