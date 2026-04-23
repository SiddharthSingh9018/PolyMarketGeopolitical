from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    root_dir: Path = Path(".")
    output_dir: Path = Path("research_outputs")
    data_dir: Path = Path("research_data")
    dataset_file: str = "model_dataset.csv"
    frequency: str = "D"
    forecast_horizon: int = 5
    realized_window: int = 5
    probability_window: int = 7
    volume_window: int = 20
    sentiment_window: int = 5
    train_fraction: float = 0.8
    min_train_periods: int = 252
    test_periods: int = 21
    step_periods: int = 21
    max_splits: int = 6
    shap_sample_size: int = 2500
    event_window: int = 2
    poly_spike_z: float = 2.0
    vix_regime_threshold: float = 25.0
    anomaly_window: int = 60
    spike_quantile: float = 0.9
    poly_z_window: int = 20
    poly_jump_threshold: float = 0.05
    bootstrap_iterations: int = 400
    min_market_volume: float = 50_000.0
    random_state: int = 7
    asset_tickers: list[str] = field(
        default_factory=lambda: ["ITA", "LMT", "NOC", "RTX", "GD", "LHX"]
    )
    benchmark_tickers: dict[str, str] = field(
        default_factory=lambda: {
            "vix": "^VIX",
            "ovx": "^OVX",
            "wti": "CL=F",
        }
    )
    macro_lags: tuple[int, ...] = (1, 2, 5)
    keyword_patterns: tuple[str, ...] = (
        "defense",
        "military",
        "missile",
        "fighter",
        "drone",
        "weapons",
        "aerospace",
        "air force",
        "navy",
        "army",
        "pentagon",
        "nato",
        "taiwan",
        "china",
        "iran",
        "israel",
        "russia",
        "ukraine",
        "war",
        "attack",
        "strike",
        "sanction",
        "raytheon",
        "lockheed",
        "northrop",
        "general dynamics",
        "l3harris",
        "boeing",
    )
    sentiment_path: Path | None = None
    market_override_path: Path | None = None
    plot_dpi: int = 180

    @property
    def raw_dir(self) -> Path:
        return self.root_dir / self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.root_dir / self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.root_dir / self.data_dir / "processed"

    @property
    def dataset_path(self) -> Path:
        return self.processed_dir / self.dataset_file

    @property
    def plot_dir(self) -> Path:
        return self.root_dir / self.output_dir / "plots"

    @property
    def table_dir(self) -> Path:
        return self.root_dir / self.output_dir / "tables"

    @property
    def text_dir(self) -> Path:
        return self.root_dir / self.output_dir / "text"

    @property
    def markets_path(self) -> Path:
        return self.root_dir / "markets.csv"

    @property
    def trades_path(self) -> Path:
        return self.root_dir / "processed" / "trades.csv"
