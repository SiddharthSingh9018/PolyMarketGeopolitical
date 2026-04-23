from __future__ import annotations

import numpy as np
import pandas as pd

from research_pipeline.config import PipelineConfig


def build_asset_panel(prices: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    panel = prices.copy().sort_values(["asset", "date"])
    panel["log_return"] = panel.groupby("asset")["close"].transform(
        lambda series: np.log(series / series.shift(1))
    )
    panel["realized_volatility"] = panel.groupby("asset")["log_return"].transform(
        lambda series: np.sqrt(series.pow(2).rolling(config.realized_window).mean())
    )
    panel["target_forward_volatility"] = panel.groupby("asset")["log_return"].transform(
        lambda series: np.sqrt(series.pow(2)[::-1].rolling(config.forecast_horizon).mean()[::-1].shift(-1))
    )
    panel["volume_change"] = panel.groupby("asset")["volume"].pct_change()
    panel["range_pct"] = (panel["high"] - panel["low"]) / panel["close"]
    panel["close_to_open"] = np.log(panel["close"] / panel["open"])
    return panel


def build_feature_table(
    asset_panel: pd.DataFrame,
    polymarket: pd.DataFrame,
    macro: pd.DataFrame,
    gpr: pd.DataFrame,
    sentiment: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    features = asset_panel.merge(polymarket, on="date", how="left")
    features = features.merge(macro, on="date", how="left")
    features = features.merge(gpr, on="date", how="left")
    features = features.merge(sentiment, on="date", how="left")

    for lag in config.macro_lags:
        for column in ["vix", "oil_volatility_proxy", "gpr", "wti_price"]:
            features[f"{column}_lag_{lag}"] = features[column].shift(lag)

    features["sentiment_rolling_mean"] = features["sentiment"].rolling(
        config.sentiment_window
    ).mean()
    features["sentiment_change"] = features["sentiment"].diff()

    fill_zero_columns = [
        "poly_probability_change",
        "poly_probability_volatility",
        "poly_volume_zscore",
        "poly_order_imbalance",
        "sentiment_change",
    ]
    for column in fill_zero_columns:
        features[column] = features[column].fillna(0.0)

    features["poly_probability_level"] = features["poly_probability_level"].ffill(limit=1)
    features["poly_daily_volume"] = features["poly_daily_volume"].fillna(0.0)
    features["poly_trade_count"] = features["poly_trade_count"].fillna(0.0)
    features["poly_market_count"] = features["poly_market_count"].fillna(0.0)
    features["sentiment"] = features["sentiment"].fillna(0.0)
    features["sentiment_rolling_mean"] = features["sentiment_rolling_mean"].fillna(0.0)
    features["regime"] = features["regime"].astype("string").fillna("unclassified")

    columns_to_drop = ["open", "high", "low", "close", "volume"]
    features = features.drop(columns=columns_to_drop)
    features = features.dropna(subset=["target_forward_volatility", "log_return", "realized_volatility"])
    return features.sort_values(["date", "asset"]).reset_index(drop=True)
