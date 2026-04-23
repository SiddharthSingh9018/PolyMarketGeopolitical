from __future__ import annotations

import json

from research_pipeline.config import PipelineConfig
from research_pipeline.data_loading import collect_frame, scan_panel
from research_pipeline.spike_evaluation import (
    aggregate_metrics,
    bootstrap_table,
    evaluate_spike_models,
    event_study_table,
    regime_train_table,
    save_plots,
    shap_outputs,
    summary_text,
)
from research_pipeline.spike_feature_engineering import build_spike_panel


def run_spike_pipeline(config: PipelineConfig | None = None) -> dict[str, object]:
    config = config or PipelineConfig()
    for directory in [config.plot_dir, config.table_dir, config.text_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    base_lf = scan_panel(config)
    spike_lf = build_spike_panel(base_lf, config)
    metrics_raw, predictions, regime_train_raw, artifacts = evaluate_spike_models(spike_lf, config)
    metrics = aggregate_metrics(metrics_raw)
    bootstrap = bootstrap_table(predictions, config)
    regime_metrics = regime_train_table(predictions)
    regime_train_metrics = aggregate_metrics(regime_train_raw) if not regime_train_raw.empty else regime_train_raw
    shap_importance = shap_outputs(artifacts, config)
    event_panel = collect_frame(spike_lf, ["date", "poly_probability_change_lag_1", "spike_top10"])
    event_table = event_study_table(event_panel, "spike_top10", config)
    save_plots(predictions, metrics, event_table, config)

    metrics.to_csv(config.table_dir / "spike_model_comparison.csv", index=False)
    bootstrap.to_csv(config.table_dir / "spike_bootstrap_results.csv", index=False)
    regime_metrics.to_csv(config.table_dir / "spike_regime_comparison.csv", index=False)
    regime_train_metrics.to_csv(config.table_dir / "spike_regime_train_comparison.csv", index=False)
    predictions.to_csv(config.table_dir / "spike_predictions.csv", index=False)
    shap_importance.to_csv(config.table_dir / "spike_shap_importance.csv", index=False)
    event_table.to_csv(config.table_dir / "spike_event_study.csv", index=False)

    summary = summary_text(metrics, bootstrap, shap_importance)
    (config.text_dir / "spike_final_summary.txt").write_text(summary, encoding="utf-8")
    (config.text_dir / "spike_manifest.json").write_text(
        json.dumps(
            {
                "dataset_path": str(config.dataset_path),
                "anomaly_window": config.anomaly_window,
                "spike_quantile": config.spike_quantile,
                "bootstrap_iterations": config.bootstrap_iterations,
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
        "bootstrap": bootstrap,
        "regime_metrics": regime_metrics,
        "regime_train_metrics": regime_train_metrics,
        "event_table": event_table,
        "shap_importance": shap_importance,
        "summary": summary,
    }
