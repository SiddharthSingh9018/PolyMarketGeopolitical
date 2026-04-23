from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl

from research_pipeline.config import PipelineConfig


def scan_panel(config: PipelineConfig) -> pl.LazyFrame:
    path = config.dataset_path
    if not path.exists():
        raise FileNotFoundError(f"Panel dataset not found at {path}")

    if path.suffix == ".parquet":
        lf = pl.scan_parquet(path)
    else:
        lf = pl.scan_csv(path, infer_schema_length=1000)

    return lf.with_columns(
        pl.col("date").str.strptime(pl.Date, strict=False),
        pl.col("asset").cast(pl.Utf8),
        pl.col("regime").cast(pl.Utf8),
    )


def collect_dates(lf: pl.LazyFrame) -> list[pd.Timestamp]:
    dates = (
        lf.select(pl.col("date"))
        .unique()
        .sort("date")
        .collect()
        .get_column("date")
        .to_list()
    )
    return [pd.Timestamp(date) for date in dates if date is not None]


def collect_slice(
    lf: pl.LazyFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    columns: list[str],
) -> pd.DataFrame:
    data = (
        lf.filter(pl.col("date") >= start_date.date())
        .filter(pl.col("date") <= end_date.date())
        .select(columns)
        .collect()
    )
    frame = pd.DataFrame(data.to_dicts())
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    if "asset" in frame.columns:
        frame["asset"] = frame["asset"].astype("category")
    return frame.sort_values(["date", "asset"]).reset_index(drop=True)


def write_lazy_snapshot(lf: pl.LazyFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lf.collect().write_parquet(path)


def collect_frame(lf: pl.LazyFrame, columns: list[str]) -> pd.DataFrame:
    data = lf.select(columns).collect()
    frame = pd.DataFrame(data.to_dicts())
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame
