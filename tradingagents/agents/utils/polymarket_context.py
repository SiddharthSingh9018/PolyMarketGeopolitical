from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import polars as pl


DEFAULT_DATASET_PATH = (
    "C:/Users/sidme/OneDrive/Desktop/Obsidian/ZettelKastan/ZettelKasten/poly_data/"
    "research_data/processed/model_dataset.csv"
)


def get_polymarket_dataset_path() -> Path:
    raw = os.getenv("POLYMARKET_DATASET_PATH", DEFAULT_DATASET_PATH)
    return Path(raw)


def _to_float(row: Dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value or 0.0)


def _format_rows(rows: List[Dict[str, Any]]) -> str:
    lines = []
    for row in rows:
        parts = [
            f"date={row.get('date')}",
            f"poly_probability_level={row.get('poly_probability_level')}",
            f"poly_probability_change={row.get('poly_probability_change')}",
            f"poly_volume_zscore={row.get('poly_volume_zscore')}",
            f"regime={row.get('regime')}",
            f"vix={row.get('vix')}",
            f"gpr={row.get('gpr')}",
            f"sentiment={row.get('sentiment')}",
        ]
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def build_polymarket_context(ticker: str, trade_date: str, lookback_rows: int = 8) -> str:
    dataset_path = get_polymarket_dataset_path()
    if not dataset_path.exists():
        return (
            "Polymarket dataset unavailable. "
            f"Expected file at `{dataset_path}`. "
            "Proceed without Polymarket evidence and say so explicitly."
        )

    as_of = date.fromisoformat(trade_date)
    frame = (
        pl.scan_csv(dataset_path)
        .with_columns(pl.col("date").str.to_date(strict=False))
        .filter((pl.col("asset") == ticker) & (pl.col("date") <= pl.lit(as_of)))
        .sort("date")
        .collect()
    )
    if frame.is_empty():
        return (
            f"No Polymarket research rows found for ticker `{ticker}` up to `{trade_date}`. "
            "Proceed without Polymarket evidence and say so explicitly."
        )

    window = frame.tail(lookback_rows)
    row = window.tail(1).to_dicts()[0]

    prob_change = _to_float(row, "poly_probability_change")
    prob_level = row.get("poly_probability_level")
    volume_z = _to_float(row, "poly_volume_zscore")
    vix = _to_float(row, "vix")
    gpr = _to_float(row, "gpr")
    sentiment_change = _to_float(row, "sentiment_change")

    derived_flags = {
        "poly_jump": abs(prob_change) >= 0.08,
        "poly_extreme": (prob_level or 0.0) <= 0.2 or (prob_level or 0.0) >= 0.8 if prob_level is not None else False,
        "poly_volume_spike": volume_z >= 1.5,
        "high_vix": vix >= 25.0,
        "elevated_gpr": gpr >= 100.0,
        "sentiment_reversal": abs(sentiment_change) >= 0.02,
    }

    snapshot_lines = [
        f"- date: {row.get('date')}",
        f"- poly_probability_level: {row.get('poly_probability_level')}",
        f"- poly_probability_change: {row.get('poly_probability_change')}",
        f"- poly_probability_volatility: {row.get('poly_probability_volatility')}",
        f"- poly_order_imbalance: {row.get('poly_order_imbalance')}",
        f"- poly_trade_count: {row.get('poly_trade_count')}",
        f"- poly_volume_zscore: {row.get('poly_volume_zscore')}",
        f"- poly_daily_volume: {row.get('poly_daily_volume')}",
        f"- poly_market_count: {row.get('poly_market_count')}",
        f"- regime: {row.get('regime')}",
        f"- vix: {row.get('vix')}",
        f"- oil_volatility_proxy: {row.get('oil_volatility_proxy')}",
        f"- wti_price: {row.get('wti_price')}",
        f"- gpr: {row.get('gpr')}",
        f"- sentiment: {row.get('sentiment')}",
        f"- sentiment_change: {row.get('sentiment_change')}",
    ]

    flag_lines = [f"- {key}: {value}" for key, value in derived_flags.items()]

    return "\n".join(
        [
            "Polymarket and geopolitical panel context:",
            *snapshot_lines,
            "",
            "Derived flags:",
            *flag_lines,
            "",
            "Recent rows:",
            _format_rows(window.select(
                [
                    "date",
                    "poly_probability_level",
                    "poly_probability_change",
                    "poly_volume_zscore",
                    "regime",
                    "vix",
                    "gpr",
                    "sentiment",
                ]
            ).to_dicts()),
        ]
    )
