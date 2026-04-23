from __future__ import annotations

import json

import pandas as pd
from matplotlib import pyplot as plt

from research_pipeline.config import PipelineConfig
from research_pipeline.data import (
    build_polymarket_panel,
    download_gpr_series,
    download_macro_series,
    download_market_series,
    ensure_polymarket_inputs,
    load_markets,
    load_sentiment_series,
    prepare_directories,
    save_data_manifest,
    save_market_selection,
    select_relevant_markets,
)
from research_pipeline.features import build_asset_panel, build_feature_table
from research_pipeline.modeling import create_shap_outputs, run_ablation


def _save_metrics(metrics: pd.DataFrame, predictions: pd.DataFrame, config: PipelineConfig) -> None:
    metrics.to_csv(config.table_dir / "evaluation_metrics.csv", index=False)
    predictions.to_csv(config.table_dir / "test_predictions.csv", index=False)

    pivot = metrics.pivot(index="segment", columns="model", values="rmse")
    ax = pivot.plot(kind="bar", figsize=(10, 5), rot=0, title="RMSE by Model and Regime")
    ax.set_ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(config.plot_dir / "rmse_by_model_and_regime.png", dpi=config.plot_dpi, bbox_inches="tight")
    plt.close()


def _write_interpretation(
    metrics: pd.DataFrame, shap_importance: pd.DataFrame, sentiment_source: str, config: PipelineConfig
) -> str:
    overall = metrics.loc[metrics["segment"].eq("overall")].sort_values("rmse")
    best_model = overall.iloc[0]
    baseline = overall.loc[overall["model"].eq("A_baseline")].iloc[0]
    improvement = baseline["rmse"] - best_model["rmse"]

    regime_rows = metrics.loc[metrics["model"].eq(best_model["model"]) & metrics["segment"].ne("overall")]
    best_regime = regime_rows.sort_values("rmse").iloc[0]["segment"] if not regime_rows.empty else "n/a"
    worst_regime = regime_rows.sort_values("rmse").iloc[-1]["segment"] if not regime_rows.empty else "n/a"

    top_poly = shap_importance.loc[
        shap_importance["feature"].str.contains("poly_probability", case=False, regex=False)
    ]
    shap_message = "SHAP does not elevate Polymarket among the top drivers."
    if not top_poly.empty and top_poly.iloc[0]["mean_abs_shap"] >= shap_importance["mean_abs_shap"].median():
        shap_message = "SHAP shows Polymarket probability features contribute materially rather than acting as redundant noise."

    interpretation = {
        "does_polymarket_improve_forecasting": bool(best_model["model"] != "A_baseline" and improvement > 0),
        "best_model": best_model["model"],
        "baseline_rmse": round(float(baseline["rmse"]), 6),
        "best_rmse": round(float(best_model["rmse"]), 6),
        "rmse_improvement_vs_baseline": round(float(improvement), 6),
        "regime_summary": {
            "best_regime": best_regime,
            "worst_regime": worst_regime,
        },
        "sentiment_source": sentiment_source,
        "shap_conclusion": shap_message,
    }

    text = (
        "Does Polymarket improve forecasting?\n"
        f"- Best model: {interpretation['best_model']}\n"
        f"- Baseline RMSE: {interpretation['baseline_rmse']}\n"
        f"- Best RMSE: {interpretation['best_rmse']}\n"
        f"- RMSE improvement vs baseline: {interpretation['rmse_improvement_vs_baseline']}\n\n"
        "Under what conditions?\n"
        f"- Lowest-error regime for the best model: {interpretation['regime_summary']['best_regime']}\n"
        f"- Highest-error regime for the best model: {interpretation['regime_summary']['worst_regime']}\n\n"
        "Does SHAP confirm contribution or redundancy?\n"
        f"- {interpretation['shap_conclusion']}\n"
        f"- Sentiment source used in this run: {sentiment_source}\n"
    )

    (config.text_dir / "interpretation.txt").write_text(text, encoding="utf-8")
    (config.text_dir / "interpretation.json").write_text(json.dumps(interpretation, indent=2), encoding="utf-8")
    return text


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, object]:
    config = config or PipelineConfig()
    prepare_directories(config)

    bootstrap_meta = ensure_polymarket_inputs(config)
    markets = load_markets(config)
    selected_markets = select_relevant_markets(markets, config)
    if selected_markets.empty:
        raise ValueError("No relevant Polymarket markets matched the configured keyword and volume filter.")

    save_market_selection(selected_markets, config)
    polymarket = build_polymarket_panel(config, selected_markets)
    prices = download_market_series(config, config.asset_tickers)
    macro = download_macro_series(config)
    gpr = download_gpr_series(config)
    sentiment, sentiment_source = load_sentiment_series(config, prices, macro)
    save_data_manifest(config, bootstrap_meta, sentiment_source, selected_markets)

    asset_panel = build_asset_panel(prices, config)
    feature_table = build_feature_table(asset_panel, polymarket, macro, gpr, sentiment, config)
    feature_table.to_csv(config.processed_dir / "model_dataset.csv", index=False)

    model_runs, metrics, predictions = run_ablation(feature_table, config)
    _save_metrics(metrics, predictions, config)
    shap_importance = create_shap_outputs(model_runs[-1], config)
    shap_importance.to_csv(config.table_dir / "shap_importance.csv", index=False)
    interpretation_text = _write_interpretation(metrics, shap_importance, sentiment_source, config)

    return {
        "metrics": metrics,
        "predictions": predictions,
        "shap_importance": shap_importance,
        "interpretation": interpretation_text,
        "selected_market_count": selected_markets["id"].nunique(),
    }
