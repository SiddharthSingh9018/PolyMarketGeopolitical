from __future__ import annotations

import polars as pl

from research_pipeline.config import PipelineConfig

SPIKE_LABELS = ["spike_roll", "spike_top10"]

BASE_FEATURES = [
    "asset",
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
    "poly_shock_lag_1",
    "poly_jump_lag_1",
]


def build_spike_panel(lf: pl.LazyFrame, config: PipelineConfig) -> pl.LazyFrame:
    numeric_columns = [
        "log_return",
        "realized_volatility",
        "target_forward_volatility",
        "volume_change",
        "range_pct",
        "close_to_open",
        "vix",
        "oil_volatility_proxy",
        "wti_price",
        "gpr",
        "sentiment",
        "poly_probability_level",
        "poly_probability_change",
        "poly_probability_volatility",
        "poly_order_imbalance",
        "poly_trade_count",
        "poly_volume_zscore",
    ]
    lf = lf.with_columns(pl.col(column).cast(pl.Float64, strict=False) for column in numeric_columns)

    asset_lf = lf.sort(["asset", "date"]).with_columns(
        pl.col("asset").cast(pl.Categorical),
        pl.col("realized_volatility").shift(1).over("asset").alias("rv_lag_1"),
        pl.col("realized_volatility").shift(2).over("asset").alias("rv_lag_2"),
        pl.col("realized_volatility").shift(5).over("asset").alias("rv_lag_5"),
        pl.col("log_return").shift(1).over("asset").alias("ret_lag_1"),
        pl.col("log_return").shift(2).over("asset").alias("ret_lag_2"),
        pl.col("log_return").shift(5).over("asset").alias("ret_lag_5"),
        pl.col("target_forward_volatility").shift(1).rolling_mean(config.anomaly_window).over("asset").alias("target_hist_mean"),
        pl.col("target_forward_volatility").shift(1).rolling_std(config.anomaly_window).over("asset").alias("target_hist_std"),
        pl.col("target_forward_volatility").quantile(config.spike_quantile).over("asset").alias("target_top10_cutoff"),
    ).with_columns(
        (
            pl.col("target_forward_volatility")
            > (pl.col("target_hist_mean") + 2.0 * pl.col("target_hist_std"))
        )
        .cast(pl.Int8)
        .alias("spike_roll"),
        (pl.col("target_forward_volatility") >= pl.col("target_top10_cutoff")).cast(pl.Int8).alias("spike_top10"),
    )

    date_lf = (
        lf.select(
            [
                "date",
                "regime",
                "vix",
                "oil_volatility_proxy",
                "wti_price",
                "gpr",
                "sentiment",
                "poly_probability_level",
                "poly_probability_change",
                "poly_probability_volatility",
                "poly_order_imbalance",
                "poly_trade_count",
                "poly_volume_zscore",
            ]
        )
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
            (
                (pl.col("poly_probability_change") - pl.col("poly_probability_change").shift(1).rolling_mean(config.poly_z_window))
                / pl.col("poly_probability_change").shift(1).rolling_std(config.poly_z_window)
            ).alias("poly_shock"),
            (pl.col("poly_probability_change").abs() > config.poly_jump_threshold).cast(pl.Int8).alias("poly_jump"),
        )
    )

    lag_exprs = []
    for column in [
        "vix",
        "oil_volatility_proxy",
        "wti_price",
        "gpr",
        "poly_probability_level",
        "poly_probability_change",
        "poly_probability_volatility",
        "poly_order_imbalance",
        "poly_trade_count",
        "poly_volume_zscore",
        "poly_shock",
        "poly_jump",
    ]:
        for lag in (1, 2, 5):
            lag_exprs.append(pl.col(column).shift(lag).alias(f"{column}_lag_{lag}"))

    date_lf = date_lf.with_columns(lag_exprs).select(
        ["date", "regime", "vix_regime", "sentiment_lag_1", "sentiment_change", "sentiment_rolling_mean"]
        + [f"{column}_lag_{lag}" for column in [
            "vix",
            "oil_volatility_proxy",
            "wti_price",
            "gpr",
            "poly_probability_level",
            "poly_probability_change",
            "poly_probability_volatility",
            "poly_order_imbalance",
            "poly_trade_count",
            "poly_volume_zscore",
            "poly_shock",
            "poly_jump",
        ] for lag in (1, 2, 5)]
    )

    model_lf = asset_lf.drop(
        [
            "regime",
            "vix",
            "oil_volatility_proxy",
            "wti_price",
            "gpr",
            "sentiment",
            "poly_probability_level",
            "poly_probability_change",
            "poly_probability_volatility",
            "poly_order_imbalance",
            "poly_trade_count",
            "poly_volume_zscore",
        ]
    ).join(date_lf, on="date", how="left")

    fill_zero = [
        "sentiment_change",
        "poly_probability_level_lag_1",
        "poly_probability_level_lag_2",
        "poly_probability_level_lag_5",
        "poly_probability_change_lag_1",
        "poly_probability_change_lag_2",
        "poly_probability_change_lag_5",
        "poly_probability_volatility_lag_1",
        "poly_probability_volatility_lag_2",
        "poly_probability_volatility_lag_5",
        "poly_order_imbalance_lag_1",
        "poly_order_imbalance_lag_2",
        "poly_order_imbalance_lag_5",
        "poly_trade_count_lag_1",
        "poly_trade_count_lag_2",
        "poly_trade_count_lag_5",
        "poly_volume_zscore_lag_1",
        "poly_volume_zscore_lag_2",
        "poly_volume_zscore_lag_5",
        "poly_shock_lag_1",
        "poly_shock_lag_2",
        "poly_shock_lag_5",
        "poly_jump_lag_1",
        "poly_jump_lag_2",
        "poly_jump_lag_5",
    ]

    exprs = [
        pl.col("regime").fill_null("unclassified"),
        pl.col("vix_regime").fill_null("low_vix"),
    ]
    float_zero = [
        column
        for column in fill_zero
        if column not in {"poly_jump_lag_1", "poly_jump_lag_2", "poly_jump_lag_5"}
    ]
    int_zero = ["poly_jump_lag_1", "poly_jump_lag_2", "poly_jump_lag_5"]
    exprs.extend(pl.col(column).fill_nan(0.0).fill_null(0.0) for column in float_zero)
    exprs.extend(pl.col(column).fill_null(0) for column in int_zero)
    return model_lf.with_columns(exprs)


def feature_sets() -> dict[str, list[str]]:
    return {
        "A_base_macro": BASE_FEATURES + MACRO_FEATURES,
        "B_base_macro_sentiment": BASE_FEATURES + MACRO_FEATURES + SENTIMENT_FEATURES,
        "C_full": BASE_FEATURES + MACRO_FEATURES + SENTIMENT_FEATURES + POLY_FEATURES,
    }


def required_columns() -> list[str]:
    columns = set(BASE_FEATURES + MACRO_FEATURES + SENTIMENT_FEATURES + POLY_FEATURES)
    columns.update(["date", "asset", "regime", "vix_regime", "realized_volatility", "spike_roll", "spike_top10"])
    return sorted(columns)
