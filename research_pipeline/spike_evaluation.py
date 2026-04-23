from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import shap
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from research_pipeline.config import PipelineConfig
from research_pipeline.data_loading import collect_dates, collect_frame, collect_slice
from research_pipeline.spike_feature_engineering import SPIKE_LABELS, feature_sets, required_columns
from research_pipeline.spike_models import fit_predict_proba, transformed_matrix


@dataclass
class RollingSplit:
    split_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def rolling_splits(dates: list[pd.Timestamp], config: PipelineConfig) -> list[RollingSplit]:
    splits: list[RollingSplit] = []
    split_id = 0
    cursor = config.min_train_periods
    while cursor + config.test_periods <= len(dates) and len(splits) < config.max_splits:
        splits.append(
            RollingSplit(
                split_id=split_id,
                train_start=dates[0],
                train_end=dates[cursor - 1],
                test_start=dates[cursor],
                test_end=dates[cursor + config.test_periods - 1],
            )
        )
        split_id += 1
        cursor += config.step_periods
    return splits


def tolerant_match_score(frame: pd.DataFrame, label_name: str, tolerance_days: int = 1) -> tuple[float, float, float]:
    matched_actual: set[tuple[str, pd.Timestamp]] = set()
    tp = 0
    preds = frame.loc[frame["pred_label"].eq(1), ["asset", "date"]].sort_values(["asset", "date"])
    actuals = frame.loc[frame[label_name].eq(1), ["asset", "date"]].sort_values(["asset", "date"])
    actual_map: dict[str, list[pd.Timestamp]] = {
        asset: group["date"].tolist() for asset, group in actuals.groupby("asset", observed=False)
    }

    for _, row in preds.iterrows():
        asset = row["asset"]
        pred_date = row["date"]
        candidates = actual_map.get(asset, [])
        best = None
        best_dist = None
        for actual_date in candidates:
            key = (asset, actual_date)
            if key in matched_actual:
                continue
            distance = abs((actual_date - pred_date).days)
            if distance <= tolerance_days and (best_dist is None or distance < best_dist):
                best = actual_date
                best_dist = distance
        if best is not None:
            matched_actual.add((asset, best))
            tp += 1

    pred_pos = int(len(preds))
    actual_pos = int(len(actuals))
    precision = tp / pred_pos if pred_pos else 0.0
    recall = tp / actual_pos if actual_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def metric_row(frame: pd.DataFrame, label_name: str, model_name: str, estimator_name: str, segment: str) -> dict[str, object]:
    y_true = frame[label_name].to_numpy()
    y_prob = frame["pred_prob"].to_numpy()
    y_pred = frame["pred_label"].to_numpy()
    precision_tol, recall_tol, f1_tol = tolerant_match_score(frame, label_name)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_prob) if np.unique(y_true).size > 1 else np.nan
    return {
        "model": model_name,
        "estimator": estimator_name,
        "label_type": label_name,
        "segment": segment,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "precision_tol": precision_tol,
        "recall_tol": recall_tol,
        "f1_tol": f1_tol,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_obs": int(len(frame)),
    }


def evaluate_spike_models(
    lf: pl.LazyFrame, config: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    dates = collect_dates(lf)
    splits = rolling_splits(dates, config)
    feature_map = feature_sets()
    columns = required_columns()
    estimators = ["logistic", "random_forest", "lightgbm"]

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    regime_train_rows: list[dict[str, object]] = []
    artifacts: dict[str, object] = {}

    for split in splits:
        train_df = collect_slice(lf, split.train_start, split.train_end, columns)
        test_df = collect_slice(lf, split.test_start, split.test_end, columns)
        train_df = train_df.dropna(subset=["asset", "realized_volatility"]).copy()
        test_df = test_df.dropna(subset=["asset", "realized_volatility"]).copy()

        for label_name in SPIKE_LABELS:
            train_label = train_df.dropna(subset=[label_name]).copy()
            test_label = test_df.dropna(subset=[label_name]).copy()
            if train_label.empty or test_label.empty:
                continue
            if train_label[label_name].nunique() < 2 or test_label[label_name].nunique() < 2:
                continue

            for model_name, feature_names in feature_map.items():
                for estimator_name in estimators:
                    pipeline, pred_prob = fit_predict_proba(
                        train_label,
                        test_label,
                        feature_names,
                        label_name,
                        estimator_name,
                        config,
                    )
                    pred_df = test_label[
                        ["date", "asset", "regime", "vix_regime", label_name]
                    ].copy()
                    pred_df["pred_prob"] = pred_prob
                    pred_df["pred_label"] = (pred_prob >= 0.5).astype(int)
                    pred_df["model"] = model_name
                    pred_df["estimator"] = estimator_name
                    pred_df["label_type"] = label_name
                    pred_df["split_id"] = split.split_id
                    prediction_frames.append(pred_df)

                    metric_rows.append(metric_row(pred_df, label_name, model_name, estimator_name, "overall"))
                    for column in ["vix_regime", "regime"]:
                        for segment, segment_df in pred_df.groupby(column):
                            if segment_df.empty or segment_df[label_name].nunique() < 2:
                                continue
                            metric_rows.append(
                                metric_row(
                                    segment_df,
                                    label_name,
                                    model_name,
                                    estimator_name,
                                    f"{column}:{segment}",
                                )
                            )

                    for regime_name in ["low_vix", "high_vix"]:
                        train_regime = train_label.loc[train_label["vix_regime"].eq(regime_name)].copy()
                        test_regime = test_label.loc[test_label["vix_regime"].eq(regime_name)].copy()
                        if (
                            train_regime.empty
                            or test_regime.empty
                            or train_regime[label_name].nunique() < 2
                            or test_regime[label_name].nunique() < 2
                        ):
                            continue
                        regime_pipeline, regime_prob = fit_predict_proba(
                            train_regime,
                            test_regime,
                            feature_names,
                            label_name,
                            estimator_name,
                            config,
                        )
                        regime_df = test_regime[["date", "asset", "vix_regime", label_name]].copy()
                        regime_df["pred_prob"] = regime_prob
                        regime_df["pred_label"] = (regime_prob >= 0.5).astype(int)
                        regime_train_rows.append(
                            metric_row(
                                regime_df,
                                label_name,
                                model_name,
                                estimator_name,
                                f"train_regime:{regime_name}",
                            )
                        )

                    if label_name == "spike_top10" and model_name == "C_full" and estimator_name == "lightgbm":
                        artifacts["shap_pipeline"] = pipeline
                        artifacts["shap_features"] = feature_names
                        artifacts["shap_frame"] = test_label.copy()
                        artifacts["label_name"] = label_name

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    regime_train = pd.DataFrame(regime_train_rows)
    return metrics, predictions, regime_train, artifacts


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["label_type", "model", "estimator", "segment"], as_index=False)
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            roc_auc=("roc_auc", "mean"),
            precision_tol=("precision_tol", "mean"),
            recall_tol=("recall_tol", "mean"),
            f1_tol=("f1_tol", "mean"),
            tn=("tn", "sum"),
            fp=("fp", "sum"),
            fn=("fn", "sum"),
            tp=("tp", "sum"),
            n_obs=("n_obs", "sum"),
        )
    )


def bootstrap_metric_diff(
    frame: pd.DataFrame,
    left_model: str,
    right_model: str,
    label_name: str,
    metric_name: str,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    work = frame.loc[frame["model"].isin([left_model, right_model])].copy()
    pivot = work.pivot_table(
        index=["date", "asset", "split_id"],
        columns="model",
        values=["pred_prob", "pred_label", label_name],
        aggfunc="first",
        observed=False,
    )
    pivot.columns = ["__".join(map(str, col)) for col in pivot.columns]
    pivot = pivot.dropna().reset_index(drop=True)
    if pivot.empty:
        return np.nan, np.nan, np.nan

    y_true = pivot[f"{label_name}__{left_model}"].to_numpy().astype(int)
    left_prob = pivot[f"pred_prob__{left_model}"].to_numpy()
    right_prob = pivot[f"pred_prob__{right_model}"].to_numpy()
    left_pred = pivot[f"pred_label__{left_model}"].to_numpy().astype(int)
    right_pred = pivot[f"pred_label__{right_model}"].to_numpy().astype(int)

    def score(metric: str, truth: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> float:
        if metric == "f1":
            return f1_score(truth, pred, zero_division=0)
        return roc_auc_score(truth, prob) if np.unique(truth).size > 1 else np.nan

    base_diff = score(metric_name, y_true, right_pred, right_prob) - score(metric_name, y_true, left_pred, left_prob)
    n_obs = len(y_true)
    for _ in range(iterations):
        idx = rng.integers(0, n_obs, n_obs)
        truth = y_true[idx]
        if np.unique(truth).size < 2:
            continue
        diff = score(metric_name, truth, right_pred[idx], right_prob[idx]) - score(metric_name, truth, left_pred[idx], left_prob[idx])
        diffs.append(diff)

    if not diffs:
        return base_diff, np.nan, np.nan
    low, high = np.quantile(diffs, [0.025, 0.975])
    return float(base_diff), float(low), float(high)


def bootstrap_table(predictions: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (label_name, estimator_name), frame in predictions.groupby(["label_type", "estimator"]):
        for left_model, right_model in [("A_base_macro", "C_full"), ("B_base_macro_sentiment", "C_full")]:
            for metric_name in ["f1", "roc_auc"]:
                diff, low, high = bootstrap_metric_diff(
                    frame,
                    left_model,
                    right_model,
                    label_name,
                    metric_name,
                    config.bootstrap_iterations,
                    config.random_state,
                )
                rows.append(
                    {
                        "label_type": label_name,
                        "estimator": estimator_name,
                        "comparison": f"{left_model} vs {right_model}",
                        "metric": metric_name,
                        "diff_right_minus_left": diff,
                        "ci_low": low,
                        "ci_high": high,
                        "significant": bool(pd.notna(low) and pd.notna(high) and (low > 0 or high < 0)),
                    }
                )
    return pd.DataFrame(rows)


def regime_train_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (label_name, regime, model_name, estimator_name), frame in predictions.groupby(
        ["label_type", "vix_regime", "model", "estimator"]
    ):
        if frame[label_name].nunique() < 2:
            continue
        rows.append(metric_row(frame, label_name, model_name, estimator_name, regime))
    return pd.DataFrame(rows)


def event_study_table(full_panel: pd.DataFrame, label_name: str, config: PipelineConfig) -> pd.DataFrame:
    panel = (
        full_panel[["date", "poly_probability_change_lag_1", label_name]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    signal = panel["poly_probability_change_lag_1"]
    zscore = (signal - signal.mean()) / signal.std(ddof=0)
    event_dates = panel.loc[zscore.abs() >= config.poly_spike_z, "date"]

    rows: list[dict[str, object]] = []
    for event_date in event_dates:
        anchor = panel.index[panel["date"].eq(event_date)][0]
        for offset in range(-config.event_window, config.event_window + 1):
            idx = anchor + offset
            if idx < 0 or idx >= len(panel):
                continue
            rows.append(
                {
                    "event_date": event_date,
                    "offset": offset,
                    "spike_rate": panel.iloc[idx][label_name],
                }
            )
    event_df = pd.DataFrame(rows)
    if event_df.empty:
        return pd.DataFrame(columns=["offset", "avg_spike_rate", "event_count"])
    return (
        event_df.groupby("offset", as_index=False)
        .agg(avg_spike_rate=("spike_rate", "mean"), event_count=("event_date", "nunique"))
    )


def shap_outputs(artifacts: dict[str, object], config: PipelineConfig) -> pd.DataFrame:
    if "shap_pipeline" not in artifacts:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    pipeline = artifacts["shap_pipeline"]
    feature_names = artifacts["shap_features"]
    test_frame = artifacts["shap_frame"].copy()
    sample = test_frame.sample(min(len(test_frame), config.shap_sample_size), random_state=config.random_state)
    matrix, transformed_names = transformed_matrix(pipeline, sample, feature_names)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer.shap_values(matrix)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({"feature": transformed_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, matrix, feature_names=transformed_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(config.plot_dir / "spike_shap_summary.png", dpi=config.plot_dpi, bbox_inches="tight")
    plt.close()

    target_feature = next((name for name in transformed_names if "poly_shock_lag_1" in name), None)
    if target_feature is None:
        target_feature = next((name for name in transformed_names if "poly_probability_change_lag_1" in name), None)
    if target_feature is not None:
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            target_feature,
            shap_values,
            matrix,
            feature_names=transformed_names,
            interaction_index="auto",
            show=False,
        )
        plt.tight_layout()
        plt.savefig(config.plot_dir / "spike_shap_dependence.png", dpi=config.plot_dpi, bbox_inches="tight")
        plt.close()
    return importance


def save_plots(predictions: pd.DataFrame, metrics: pd.DataFrame, event_table: pd.DataFrame, config: PipelineConfig) -> None:
    overall = metrics.loc[metrics["segment"].eq("overall") & metrics["label_type"].eq("spike_top10")].sort_values(
        ["f1", "roc_auc"], ascending=[False, False]
    )
    if overall.empty:
        overall = metrics.loc[metrics["segment"].eq("overall")].sort_values(["f1", "roc_auc"], ascending=[False, False])
    if not overall.empty:
        label_name = overall.iloc[0]["label_type"]
        best = overall.iloc[0][["model", "estimator"]].to_dict()
        best_pred = predictions.loc[
            predictions["label_type"].eq(label_name)
            & predictions["model"].eq(best["model"])
            & predictions["estimator"].eq(best["estimator"])
        ].copy()
        plot_frame = best_pred.groupby("date", as_index=False).agg(
            actual_spike=(label_name, "mean"),
            predicted_prob=("pred_prob", "mean"),
        )
        ax = plot_frame.plot(
            x="date",
            y=["actual_spike", "predicted_prob"],
            figsize=(11, 5),
            title="Predicted vs Actual Spike Intensity",
        )
        ax.set_ylabel("Rate / Probability")
        plt.tight_layout()
        plt.savefig(config.plot_dir / "spike_predicted_vs_actual.png", dpi=config.plot_dpi, bbox_inches="tight")
        plt.close()

    regime_plot = metrics.loc[
        metrics["segment"].str.startswith("vix_regime:") & metrics["label_type"].eq("spike_top10")
    ].pivot_table(index="segment", columns=["model", "estimator"], values="f1", observed=False)
    if not regime_plot.empty:
        regime_plot.plot(kind="bar", figsize=(12, 5), rot=0, title="Spike F1 by VIX Regime")
        plt.tight_layout()
        plt.savefig(config.plot_dir / "spike_regime_comparison.png", dpi=config.plot_dpi, bbox_inches="tight")
        plt.close()

    if not event_table.empty:
        ax = event_table.plot(x="offset", y="avg_spike_rate", marker="o", figsize=(8, 4), title="Spike Event Study")
        ax.set_ylabel("Average Spike Rate")
        plt.tight_layout()
        plt.savefig(config.plot_dir / "spike_event_study.png", dpi=config.plot_dpi, bbox_inches="tight")
        plt.close()


def summary_text(metrics: pd.DataFrame, bootstrap: pd.DataFrame, shap_importance: pd.DataFrame) -> str:
    preferred_label = "spike_top10"
    overall = metrics.loc[metrics["segment"].eq("overall") & metrics["label_type"].eq(preferred_label)].sort_values(
        ["f1", "roc_auc"], ascending=[False, False]
    )
    if overall.empty:
        overall = metrics.loc[metrics["segment"].eq("overall")].sort_values(["f1", "roc_auc"], ascending=[False, False])
        if overall.empty:
            return (
                "Q1: Does Polymarket improve spike prediction? NO\n"
                "Q2: Is improvement statistically significant? NO\n"
                "Q3: In which regimes does it help? insufficient_data\n"
            )
        preferred_label = str(overall.iloc[0]["label_type"])
    best = overall.iloc[0]
    c_full = overall.loc[overall["model"].eq("C_full")].sort_values(["f1", "roc_auc"], ascending=[False, False])
    c_best = c_full.iloc[0] if not c_full.empty else None

    sig_rows = bootstrap.loc[
        bootstrap["comparison"].isin(["A_base_macro vs C_full", "B_base_macro_sentiment vs C_full"])
        & bootstrap["metric"].eq("f1")
        & bootstrap["label_type"].eq(preferred_label)
    ]
    significant = bool((sig_rows["significant"]).all()) if not sig_rows.empty else False
    improve = bool(c_best is not None and best["model"] == "C_full")
    high_vix = metrics.loc[
        metrics["segment"].eq("vix_regime:high_vix") & metrics["label_type"].eq(preferred_label) & metrics["model"].eq("C_full")
    ]
    low_vix = metrics.loc[
        metrics["segment"].eq("vix_regime:low_vix") & metrics["label_type"].eq(preferred_label) & metrics["model"].eq("C_full")
    ]
    regime_text = "high_vix" if not high_vix.empty and (low_vix.empty or high_vix["f1"].max() > low_vix["f1"].max()) else "low_vix_or_none"

    poly_top = shap_importance.loc[shap_importance["feature"].str.contains("poly_", regex=False)].head(3)["feature"].tolist()
    return (
        f"Q1: Does Polymarket improve spike prediction? {'YES' if improve else 'NO'}\n"
        f"Q2: Is improvement statistically significant? {'YES' if significant else 'NO'}\n"
        f"Q3: In which regimes does it help? {regime_text}\n"
        f"- Label definition used for summary: {preferred_label}\n"
        f"- Best overall spike model: {best['model']} with {best['estimator']}\n"
        f"- Best F1: {best['f1']:.6f}\n"
        f"- Best C_full F1: {float(c_best['f1']) if c_best is not None else float('nan'):.6f}\n"
        f"- Top Polymarket SHAP features: {', '.join(poly_top) if poly_top else 'none'}\n"
    )
