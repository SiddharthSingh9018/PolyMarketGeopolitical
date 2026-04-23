# Methodology and Findings

## Research Question
This project evaluates whether decentralized prediction market signals from Polymarket add incremental value to short-horizon risk forecasting for U.S. aerospace and defense assets.

Two related tasks were studied:

1. Volatility forecasting as a regression problem.
2. Volatility spike detection as a rare-event classification problem.

Assets in the panel include `ITA`, `LMT`, `NOC`, `RTX`, `GD`, and `LHX`.

---

## Data and Pipeline Design

### Data sources
The working panel combines:

- Asset-level market variables:
  - `log_return`
  - `realized_volatility`
  - `volume_change`
  - `range_pct`
  - `close_to_open`
- Macro variables:
  - `vix`
  - `oil_volatility_proxy`
  - `wti_price`
  - `gpr`
  - `sentiment`
- Polymarket variables:
  - `poly_probability_level`
  - `poly_probability_change`
  - `poly_probability_volatility`
  - `poly_order_imbalance`
  - `poly_trade_count`
  - `poly_volume_zscore`

### Scalability choices
Because the underlying data are large, the pipeline was designed around lazy loading and split-wise materialization rather than loading the full dataset into pandas memory at once.

Implementation choices:

- Lazy scanning with Polars.
- Strict sorting by `date` and `asset`.
- Feature generation done with lag operators so only past information is used.
- Rolling time-series splits instead of random train/test sampling.

### Leakage control
The pipeline explicitly avoids forward leakage:

- Asset-level lags such as return and realized volatility are computed within each asset.
- Date-level macro and Polymarket lags are computed on unique dates, then joined back to the asset panel.
- Model fitting uses rolling windows where training always precedes test periods in calendar time.
- Missing values are handled inside model pipelines using train-time imputers rather than using future information.

---

## Feature Sets

Three nested feature sets were evaluated.

### Model A
`BASE + MACRO`

Includes lagged realized volatility, lagged returns, microstructure variables, and macro variables.

### Model B
`BASE + MACRO + SENTIMENT`

Adds sentiment-related variables.

### Model C
`BASE + MACRO + SENTIMENT + POLY`

Adds Polymarket-derived features.

For spike classification, the Polymarket block was extended with:

- `poly_shock`
- `poly_jump`

where `poly_shock` captures standardized probability changes and `poly_jump` flags large absolute changes.

---

## Modeling Strategy

## 1. Regression task
Target:

- `target_forward_volatility`

Models tested:

- Linear regression
- Random forest
- XGBoost

Evaluation:

- RMSE
- MAE
- R²
- directional accuracy
- Diebold-Mariano comparisons between benchmark and Polymarket-augmented models

## 2. Spike classification task
The same panel was reformulated as a binary classification problem.

Two spike labels were constructed:

### Spike definition 1
`spike_roll = 1` if forward volatility exceeded:

`rolling mean + 2 * rolling std`

using only historical values for each asset.

### Spike definition 2
`spike_top10 = 1` if forward volatility was in the top 10% of the asset’s distribution.

Models tested:

- Logistic regression
- Random forest classifier
- LightGBM classifier

Evaluation:

- precision
- recall
- F1
- ROC-AUC
- confusion matrix components
- tolerant event scoring with ±1 day allowance
- bootstrap confidence intervals for `F1` and `ROC-AUC`

---

## Regime Analysis

Two regime concepts were used:

- `regime` from the Polymarket probability bucket where available
- `vix_regime`, defined as:
  - `high_vix` if lagged VIX >= 25
  - `low_vix` otherwise

Performance was compared within regimes, and for the classification pipeline separate regime-specific training summaries were also saved.

---

## Explainability

SHAP was applied to the augmented tree-based models.

Outputs include:

- global SHAP importance
- dependence plots
- Polymarket-only SHAP ranking plots

This was used to test whether Polymarket variables contribute meaningful information or merely duplicate information already contained in conventional risk indicators.

---

## Findings: Regression

### Overall result
Polymarket did **not** produce the single best overall volatility forecasting model.

Best overall regression result:

- `A_base_macro + linear`
- RMSE: `0.024193`
- MAE: `0.020057`
- directional accuracy: `0.554233`

Best Polymarket-augmented regression result:

- `C_full + random_forest`
- RMSE: `0.024369`
- MAE: `0.020338`
- directional accuracy: `0.542328`

### Interpretation
This means the strongest overall out-of-sample regression model remained a traditional benchmark specification.

However, the augmented models were not uniformly dominated:

- `C_full` outperformed its matched benchmark in some estimator families.
- Diebold-Mariano comparisons suggested incremental gains in some pairwise settings.
- SHAP indicated that Polymarket variables such as:
  - `poly_probability_change_lag_1`
  - `poly_probability_level_lag_1`
  - `poly_order_imbalance_lag_1`
  had non-zero explanatory relevance.

### Regression conclusion
For continuous volatility forecasting, Polymarket features appear informative, but they did **not** beat the strongest traditional benchmark overall in this run.

---

## Findings: Spike Classification

### Main headline
The spike-classification results were more nuanced than the regression results, but the final answer remained conservative:

- `Q1: Does Polymarket improve spike prediction? NO`
- `Q2: Is the improvement statistically significant? NO`
- `Q3: In which regimes does it help? high_vix`

### Best overall classification performance
For the preferred rare-event summary label `spike_top10`, the best overall model was:

- `A_base_macro + logistic`
- Precision: `0.724221`
- Recall: `0.719823`
- F1: `0.655090`
- ROC-AUC: `0.478977`

Best Polymarket-augmented `C_full` result:

- `C_full + logistic`
- Precision: `0.719495`
- Recall: `0.724410`
- F1: `0.642701`
- ROC-AUC: `0.458593`

Best `C_full` tree result for `spike_top10`:

- `C_full + random_forest`
- F1: `0.620282`
- ROC-AUC: `0.620503`

### What the bootstrap tests showed
Bootstrap comparisons do show localized gains, but not a robust overall improvement.

Examples:

- `spike_top10`, `lightgbm`, `A_base_macro vs C_full`, `F1`
  - improvement: `+0.007072`
  - 95% CI: `[0.001337, 0.014732]`
  - statistically significant

- `spike_top10`, `logistic`, `A_base_macro vs C_full`, `F1`
  - improvement: `+0.006755`
  - 95% CI: `[-0.016348, 0.026313]`
  - not statistically significant

- `spike_top10`, `random_forest`, `A_base_macro vs C_full`, `F1`
  - improvement: `+0.004947`
  - 95% CI: `[-0.000932, 0.011917]`
  - not statistically significant

So while some estimator-specific improvements exist, they are not broad or consistent enough to support a strong claim that Polymarket reliably improves spike prediction across the board.

### Regime findings
The help from Polymarket appears more visible in high-VIX conditions than in low-VIX periods.

Examples for `spike_top10` in `high_vix`:

- `C_full + lightgbm`
  - F1: `0.731597`
- `C_full + random_forest`
  - F1: `0.728950`
- `C_full + logistic`
  - F1: `0.700957`

In `low_vix`, F1 levels were materially weaker for most models, and the Polymarket block did not clearly dominate.

This suggests that prediction-market signals may be more useful when the broader risk environment is already elevated.

---

## SHAP Findings

### Regression SHAP
Regression SHAP indicated that some Polymarket features matter, but they were not dominant enough to overturn the baseline advantage.

### Spike SHAP
For spike classification, the largest SHAP values were still mostly driven by macro and market volatility variables.

Top global spike SHAP drivers included:

- `vix_lag_1`
- `wti_price_lag_5`
- `vix_lag_2`
- `rv_lag_5`
- `oil_volatility_proxy_lag_5`

Top Polymarket SHAP features were more modest:

- `poly_volume_zscore_lag_1`
- `poly_trade_count_lag_1`
- `poly_order_imbalance_lag_1`

Interpretation:

- Polymarket features were not irrelevant.
- But they were not the dominant drivers of rare-event prediction once conventional volatility and macro signals were included.

---

## Final Interpretation

### Answer to the core question
Do decentralized prediction market probabilities improve short-horizon volatility forecasting for defense assets beyond traditional benchmarks?

**No overall.**

The best overall model in the regression task remained a non-Polymarket benchmark, and the best overall spike-classification model also remained a benchmark specification.

### More careful research conclusion
A more precise statement is:

- Polymarket variables contain information.
- They sometimes improve matched models, especially in some tree-based and high-VIX settings.
- They are more useful as conditional or regime-sensitive signals than as universally dominant predictors.
- The evidence does not support a broad claim of statistically robust overall improvement beyond traditional benchmarks.

### Practical implication
Polymarket may be more valuable as:

- a supplementary event-risk indicator,
- a high-volatility regime signal,
- or an early-warning overlay for spike detection,

rather than as a standalone replacement for macro-volatility benchmarks.

---

## Output Files

Key result tables:

- `research_outputs/tables/model_comparison.csv`
- `research_outputs/tables/diebold_mariano_results.csv`
- `research_outputs/tables/spike_model_comparison.csv`
- `research_outputs/tables/spike_bootstrap_results.csv`
- `research_outputs/tables/spike_regime_comparison.csv`
- `research_outputs/tables/spike_shap_importance.csv`

Key summaries:

- `research_outputs/text/final_summary.txt`
- `research_outputs/text/spike_final_summary.txt`

Key plots:

- `research_outputs/plots/spike_improvement_heatmaps.png`
- `research_outputs/plots/spike_bootstrap_forest.png`
- `research_outputs/plots/spike_regime_f1_bars.png`
- `research_outputs/plots/spike_shap_top15.png`
- `research_outputs/plots/spike_shap_poly_only.png`

