from __future__ import annotations

import polars as pl

from research_pipeline.config import PipelineConfig

BASE_FEATURES = [
    "asset",
    "log_return",
    "realized_volatility",
    "rv_lag_1",
    "rv_lag_2",
    "rv_lag_5",
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_5",
    "volume_change",
    "range_pct",
    "close_to_open",
]

MACRO_FEATURES = [
    "vix_lag_1",
    "vix_lag_2",
    "vix_lag_5",
    "oil_volatility_proxy_lag_1",
    "oil_volatility_proxy_lag_2",
    "oil_volatility_proxy_lag_5",
    "wti_price_lag_1",
    "wti_price_lag_2",
    "wti_price_lag_5",
    "gpr_lag_1",
    "gpr_lag_2",
    "gpr_lag_5",
]

SENTIMENT_FEATURES = [
    "sentiment_lag_1",
    "sentiment_change",
    "sentiment_rolling_mean",
]

POLY_FEATURES = [
    "poly_probability_level_lag_1",
    "poly_probability_change_lag_1",
    "poly_probability_volatility_lag_1",
    "poly_order_imbalance_lag_1",
    "poly_trade_count_lag_1",
    "poly_volume_zscore_lag_1",
]

TARGET_COLUMN = "target_forward_volatility"
REGIME_COLUMNS = ["regime", "vix_regime"]


def build_model_panel(lf: pl.LazyFrame, config: PipelineConfig) -> pl.LazyFrame:
    macro_columns = ["vix", "oil_volatility_proxy", "wti_price", "gpr"]
    poly_columns = [
        "poly_probability_level",
        "poly_probability_change",
        "poly_probability_volatility",
        "poly_order_imbalance",
        "poly_trade_count",
        "poly_volume_zscore",
    ]

    asset_lf = lf.sort(["asset", "date"]).with_columns(
        pl.col("asset").cast(pl.Categorical),
        pl.col("realized_volatility").shift(1).over("asset").alias("rv_lag_1"),
        pl.col("realized_volatility").shift(2).over("asset").alias("rv_lag_2"),
        pl.col("realized_volatility").shift(5).over("asset").alias("rv_lag_5"),
        pl.col("log_return").shift(1).over("asset").alias("ret_lag_1"),
        pl.col("log_return").shift(2).over("asset").alias("ret_lag_2"),
        pl.col("log_return").shift(5).over("asset").alias("ret_lag_5"),
    )

    date_lf = (
        lf.select(["date", "sentiment"] + macro_columns + poly_columns + ["regime"])
        .unique(subset=["date"], keep="first")
        .sort("date")
        .with_columns(
            pl.col("sentiment").shift(1).alias("sentiment_lag_1"),
            pl.col("sentiment").diff().alias("sentiment_change"),
            pl.col("sentiment").shift(1).rolling_mean(config.sentiment_window).alias("sentiment_rolling_mean"),
            pl.when(pl.col("vix").shift(1) >= config.vix_regime_threshold)
            .then(pl.lit("high_vix"))
            .otherwise(pl.lit("low_vix"))
            .alias("vix_regime"),
        )
    )

    lag_exprs = []
    for column in macro_columns + poly_columns:
        lag_exprs.append(pl.col(column).shift(1).alias(f"{column}_lag_1"))
        lag_exprs.append(pl.col(column).shift(2).alias(f"{column}_lag_2"))
        lag_exprs.append(pl.col(column).shift(5).alias(f"{column}_lag_5"))
    date_lf = date_lf.with_columns(lag_exprs).select(
        ["date", "regime", "vix_regime", "sentiment_lag_1", "sentiment_change", "sentiment_rolling_mean"]
        + [f"{column}_lag_{lag}" for column in macro_columns + poly_columns for lag in (1, 2, 5)]
    )

    model_lf = asset_lf.drop(["regime"] + macro_columns + poly_columns).join(date_lf, on="date", how="left")
    return model_lf.with_columns(
        pl.col("regime").fill_null("unclassified"),
        pl.col("vix_regime").fill_null("low_vix"),
        pl.col("sentiment_change").fill_null(0.0),
        pl.col("poly_probability_level_lag_1").fill_null(0.0),
        pl.col("poly_probability_level_lag_2").fill_null(0.0),
        pl.col("poly_probability_level_lag_5").fill_null(0.0),
        pl.col("poly_probability_change_lag_1").fill_null(0.0),
        pl.col("poly_probability_change_lag_2").fill_null(0.0),
        pl.col("poly_probability_change_lag_5").fill_null(0.0),
        pl.col("poly_probability_volatility_lag_2").fill_null(0.0),
        pl.col("poly_probability_volatility_lag_5").fill_null(0.0),
        pl.col("poly_probability_volatility_lag_1").fill_null(0.0),
        pl.col("poly_order_imbalance_lag_1").fill_null(0.0),
        pl.col("poly_order_imbalance_lag_2").fill_null(0.0),
        pl.col("poly_order_imbalance_lag_5").fill_null(0.0),
        pl.col("poly_trade_count_lag_1").fill_null(0.0),
        pl.col("poly_trade_count_lag_2").fill_null(0.0),
        pl.col("poly_trade_count_lag_5").fill_null(0.0),
        pl.col("poly_volume_zscore_lag_1").fill_null(0.0),
        pl.col("poly_volume_zscore_lag_2").fill_null(0.0),
        pl.col("poly_volume_zscore_lag_5").fill_null(0.0),
    )


def feature_sets() -> dict[str, list[str]]:
    return {
        "A_base_macro": BASE_FEATURES + MACRO_FEATURES,
        "B_base_macro_sentiment": BASE_FEATURES + MACRO_FEATURES + SENTIMENT_FEATURES,
        "C_full": BASE_FEATURES + MACRO_FEATURES + SENTIMENT_FEATURES + POLY_FEATURES,
    }


def required_columns() -> list[str]:
    columns = set(BASE_FEATURES + MACRO_FEATURES + SENTIMENT_FEATURES + POLY_FEATURES + REGIME_COLUMNS)
    columns.update(["date", "asset", TARGET_COLUMN, "realized_volatility"])
    return sorted(columns)
