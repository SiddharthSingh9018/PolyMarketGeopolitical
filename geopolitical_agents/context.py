from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

import polars as pl

from geopolitical_agents.config import AgentConfig


@dataclass
class ResearchContext:
    asset: str
    as_of_date: str
    snapshot: dict[str, Any]
    recent_rows: list[dict[str, Any]]
    derived_flags: dict[str, Any]
    source_note: Optional[str] = None

    def to_markdown(self) -> str:
        lines = [
            f"Asset: {self.asset}",
            f"As of date: {self.as_of_date}",
            "",
            "Current snapshot:",
        ]
        for key, value in self.snapshot.items():
            lines.append(f"- {key}: {value}")
        lines.extend(["", "Derived flags:"])
        for key, value in self.derived_flags.items():
            lines.append(f"- {key}: {value}")
        lines.extend(["", "Recent rows:"])
        for row in self.recent_rows:
            compact = ", ".join(f"{key}={value}" for key, value in row.items())
            lines.append(f"- {compact}")
        if self.source_note:
            lines.extend(["", "External source note:", self.source_note])
        return "\n".join(lines)


def _optional_value(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None:
        return None
    return value


def _read_source_note(path: Optional[Path], char_limit: int) -> Optional[str]:
    if not path or not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[:char_limit]


def build_research_context(
    config: AgentConfig,
    asset: str,
    as_of_date: str,
    source_note_path: Optional[Path] = None,
) -> ResearchContext:
    as_of = date.fromisoformat(as_of_date)
    scan = (
        pl.scan_csv(config.dataset_path)
        .with_columns(pl.col("date").str.to_date(strict=False))
        .filter((pl.col("asset") == asset) & (pl.col("date") <= pl.lit(as_of)))
        .sort("date")
    )
    frame = scan.collect()
    if frame.is_empty():
        raise ValueError(f"No rows found for asset={asset} up to {as_of_date}.")

    window = frame.tail(config.lookback_rows)
    last_row = window.tail(1).to_dicts()[0]

    snapshot_keys = [
        "date",
        "log_return",
        "realized_volatility",
        "volume_change",
        "range_pct",
        "close_to_open",
        "poly_probability_level",
        "poly_probability_change",
        "poly_probability_volatility",
        "poly_order_imbalance",
        "poly_trade_count",
        "poly_volume_zscore",
        "poly_daily_volume",
        "poly_market_count",
        "regime",
        "vix",
        "oil_volatility_proxy",
        "wti_price",
        "gpr",
        "sentiment",
        "sentiment_change",
        "sentiment_rolling_mean",
    ]
    snapshot = {key: _optional_value(last_row, key) for key in snapshot_keys}

    recent_rows = []
    for row in window.select(
        [
            "date",
            "log_return",
            "realized_volatility",
            "poly_probability_level",
            "poly_probability_change",
            "poly_volume_zscore",
            "regime",
            "vix",
            "gpr",
            "sentiment",
        ]
    ).to_dicts():
        recent_rows.append(row)

    prob_change = float(last_row.get("poly_probability_change") or 0.0)
    prob_level = float(last_row.get("poly_probability_level") or 0.0)
    vix = float(last_row.get("vix") or 0.0)
    volume_z = float(last_row.get("poly_volume_zscore") or 0.0)
    gpr = float(last_row.get("gpr") or 0.0)
    sentiment_change = float(last_row.get("sentiment_change") or 0.0)

    derived_flags = {
        "poly_jump": abs(prob_change) >= 0.08,
        "poly_extreme": prob_level <= 0.2 or prob_level >= 0.8,
        "poly_volume_spike": volume_z >= 1.5,
        "high_vix": vix >= 25.0,
        "elevated_gpr": gpr >= 120.0,
        "sentiment_reversal": abs(sentiment_change) >= 0.02,
        "event_driven_setup": any(
            [
                abs(prob_change) >= 0.08,
                volume_z >= 1.5,
                vix >= 25.0,
                gpr >= 120.0,
            ]
        ),
    }

    return ResearchContext(
        asset=asset,
        as_of_date=str(last_row["date"]),
        snapshot=snapshot,
        recent_rows=recent_rows,
        derived_flags=derived_flags,
        source_note=_read_source_note(source_note_path, config.source_note_char_limit)
        if config.include_source_note
        else None,
    )
