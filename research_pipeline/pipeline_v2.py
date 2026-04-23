from __future__ import annotations

import json

from research_pipeline.config import PipelineConfig
from research_pipeline.data_loading import collect_frame, scan_panel
from research_pipeline.evaluation import (
    aggregate_metrics,
    dm_table,
    evaluate_models,
    event_study_table,
    regime_model_table,
    save_plots,
    shap_outputs,
    summary_text,
)
from research_pipeline.feature_engineering import build_model_panel


def run_large_scale_pipeline(config: PipelineConfig | None = None) -> dict[str, object]:
    config = config or PipelineConfig()
    for directory in [config.plot_dir, config.table_dir, config.text_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    base_lf = scan_panel(config)
    model_lf = build_model_panel(base_lf, config)
    metrics_raw, predictions, artifacts = evaluate_models(model_lf, config)
    metrics = aggregate_metrics(metrics_raw)
    dm_results = dm_table(predictions, config)
    regime_results = regime_model_table(predictions)
    shap_importance = shap_outputs(artifacts, config)
    event_panel = collect_frame(model_lf, ["date", "poly_probability_change_lag_1", "target_forward_volatility"])
    event_table = event_study_table(event_panel, config)
    save_plots(predictions, metrics, event_table, config)

    metrics.to_csv(config.table_dir / "model_comparison.csv", index=False)
    dm_results.to_csv(config.table_dir / "diebold_mariano_results.csv", index=False)
    regime_results.to_csv(config.table_dir / "regime_model_comparison.csv", index=False)
    predictions.to_csv(config.table_dir / "rolling_predictions.csv", index=False)
    shap_importance.to_csv(config.table_dir / "shap_importance.csv", index=False)
    event_table.to_csv(config.table_dir / "event_study.csv", index=False)

    summary = summary_text(metrics, dm_results, shap_importance)
    (config.text_dir / "final_summary.txt").write_text(summary, encoding="utf-8")
    (config.text_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "dataset_path": str(config.dataset_path),
                "min_train_periods": config.min_train_periods,
                "test_periods": config.test_periods,
                "step_periods": config.step_periods,
                "max_splits": config.max_splits,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "metrics": metrics,
        "dm_results": dm_results,
        "regime_results": regime_results,
        "event_table": event_table,
        "shap_importance": shap_importance,
        "summary": summary,
    }
