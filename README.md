# Polymarket Defense Volatility Research Pipeline

This repository combines:

1. A Polymarket data ingestion stack.
2. A research pipeline for testing whether decentralized prediction market signals improve short-horizon volatility forecasting and volatility spike detection in U.S. defense-related assets.

The project is currently structured as a research-grade workflow rather than only a data-collection repo.

## Research Question

The core question is:

> Do decentralized prediction market probabilities from Polymarket improve short-horizon volatility forecasting and volatility spike prediction for defense assets beyond traditional benchmark variables?

The working conclusion in the current saved outputs is:

- not robustly for broad daily forecasting
- possibly in narrower, regime-sensitive, or event-driven settings

## Intended Research Positioning

The framing is suitable for empirical work aimed at outlets such as:

- *Journal of Prediction Markets*
- *Management Science*
- *American Political Science Review*
- *Journal of International Money and Finance*
- *Journal of Futures Markets*
- *Finance Research Letters*

The current evidence is most naturally aligned with:

- *Journal of Prediction Markets*
- *Journal of Futures Markets*
- *Finance Research Letters*

because the result is nuanced, conditional, and event-sensitive rather than a universal forecasting breakthrough.

## Repository Layers

### 1. Data ingestion
The original repo components collect:

- market metadata from the Polymarket API
- order-filled events from Goldsky
- processed trade records with price and side information

### 2. Research pipeline
The added research stack runs:

- volatility forecasting regression
- volatility spike classification
- feature-set ablation
- regime analysis
- event studies
- statistical testing
- SHAP analysis
- plot and report generation

## Repository Layout

```text
poly_data/
├── update_all.py
├── update_utils/
├── poly_utils/
├── research_pipeline/
│   ├── config.py
│   ├── data.py
│   ├── data_loading.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── evaluation.py
│   ├── pipeline.py
│   ├── pipeline_v2.py
│   ├── spike_feature_engineering.py
│   ├── spike_models.py
│   ├── spike_evaluation.py
│   ├── spike_pipeline.py
│   └── plot_extras.py
├── run_research_pipeline.py
├── run_spike_pipeline.py
├── generate_extra_plots.py
├── research_data/
└── research_outputs/
```

## Installation

This project uses [UV](https://docs.astral.sh/uv/).

```bash
uv sync
```

Optional notebook extras:

```bash
uv sync --extra dev
```

## Ingestion Workflow

To update the raw Polymarket data:

```bash
uv run python update_all.py
```

This runs:

- `update_markets()`
- `update_goldsky()`
- `process_live()`

## Research Workflows

## 1. Regression pipeline

```bash
uv run python run_research_pipeline.py
```

This runs:

- rolling time-series regression
- benchmark vs sentiment vs Polymarket feature-set comparison
- Diebold-Mariano testing
- SHAP explainability

## 2. Spike-classification pipeline

```bash
uv run python run_spike_pipeline.py
```

This runs:

- `spike_roll` classification
- `spike_top10` classification
- precision / recall / F1 / ROC-AUC evaluation
- bootstrap confidence intervals
- McNemar tests
- tolerant `±1 day` timing evaluation
- regime comparison
- SHAP explainability

## 3. Extra plots

```bash
uv run python generate_extra_plots.py
```

## Paper-Style Profile

The code now includes a paper-style configuration helper:

- `paper_2020_profile()`

This is designed to move the code closer to the study layout described in the paper draft:

- date range: `2020-01-02` to `2020-07-31`
- assets:
  - `RTX`
  - `LMT`
  - `NOC`
- 60-day train window
- 30-day held-out test window

This helper makes the code more paper-aligned, but the existing saved outputs in the repo were not all regenerated under that exact profile.

## Dataset

The main research panel is:

- [research_data/processed/model_dataset.csv](research_data/processed/model_dataset.csv)

It is a `date × asset` panel containing:

- market features:
  - `log_return`
  - `realized_volatility`
  - `target_forward_volatility`
  - `volume_change`
  - `range_pct`
  - `close_to_open`
- Polymarket features:
  - `poly_probability_level`
  - `poly_probability_change`
  - `poly_probability_volatility`
  - `poly_order_imbalance`
  - `poly_trade_count`
  - `poly_volume_zscore`
  - `poly_daily_volume`
  - `poly_market_count`
  - `regime`
- macro features:
  - `vix`
  - `oil_volatility_proxy`
  - `wti_price`
  - `gpr`
  - optional Treasury-yield support in code
- sentiment features:
  - `sentiment`
  - `sentiment_change`
  - `sentiment_rolling_mean`
- lagged controls:
  - `*_lag_1`
  - `*_lag_2`
  - `*_lag_5`

## Methodology Summary

### Regression task
Target:

- `target_forward_volatility`

Feature sets:

- `A_base_macro`
- `B_base_macro_sentiment`
- `C_full`

Estimators:

- Linear Regression
- Random Forest
- XGBoost

Metrics:

- RMSE
- MAE
- R²
- directional accuracy

Inference:

- Diebold-Mariano tests

### Spike task
Labels:

- `spike_roll`
- `spike_top10`

Additional event-sensitive Polymarket features:

- `poly_shock`
- `poly_jump`

Estimators:

- Logistic Regression
- Random Forest
- LightGBM

Metrics:

- precision
- recall
- F1
- ROC-AUC
- confusion counts
- tolerant event scoring

Inference:

- bootstrap confidence intervals
- McNemar tests

## Key Outputs

### Main writeups

- [research_outputs/text/methodology_and_findings.md](research_outputs/text/methodology_and_findings.md)
- [research_outputs/text/paper_vs_pipeline_comparison.md](research_outputs/text/paper_vs_pipeline_comparison.md)
- [research_outputs/text/final_summary.txt](research_outputs/text/final_summary.txt)
- [research_outputs/text/spike_final_summary.txt](research_outputs/text/spike_final_summary.txt)

### Main tables

- [research_outputs/tables/model_comparison.csv](research_outputs/tables/model_comparison.csv)
- [research_outputs/tables/diebold_mariano_results.csv](research_outputs/tables/diebold_mariano_results.csv)
- [research_outputs/tables/spike_model_comparison.csv](research_outputs/tables/spike_model_comparison.csv)
- [research_outputs/tables/spike_bootstrap_results.csv](research_outputs/tables/spike_bootstrap_results.csv)
- [research_outputs/tables/spike_mcnemar_results.csv](research_outputs/tables/spike_mcnemar_results.csv)
- [research_outputs/tables/spike_regime_comparison.csv](research_outputs/tables/spike_regime_comparison.csv)
- [research_outputs/tables/spike_shap_importance.csv](research_outputs/tables/spike_shap_importance.csv)

### Plots

- [research_outputs/plots](research_outputs/plots)

Useful plot files include:

- `spike_improvement_heatmaps.png`
- `spike_bootstrap_forest.png`
- `spike_regime_f1_bars.png`
- `spike_shap_top15.png`
- `spike_shap_poly_only.png`
- `spike_event_study.png`
- `shap_summary.png`
- `spike_shap_summary.png`

## Current Empirical Takeaway

### Regression
Polymarket features were informative but did not produce the single best overall continuous volatility forecast in the saved run.

### Spike detection
Polymarket features did not produce a broad, statistically robust overall improvement in spike prediction, although they appear more useful in higher-volatility or event-driven settings.

### Best high-level interpretation
Polymarket is best viewed as:

- a supplementary event-risk indicator
- a conditional regime signal
- an early-warning overlay

rather than as a universally dominant daily forecasting input.

## Notes

- Raw download caches under `research_data/raw/` are ignored by Git.
- Large source data are handled with lazy execution and split-wise materialization.
- The repo now contains both ingestion logic and research outputs.

## Troubleshooting

### Missing research outputs
Run:

```bash
uv run python run_research_pipeline.py
uv run python run_spike_pipeline.py
uv run python generate_extra_plots.py
```

### Missing raw Polymarket data
Run:

```bash
uv run python update_all.py
```

## License

Go wild with it.
