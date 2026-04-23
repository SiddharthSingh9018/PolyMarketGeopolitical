from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
import shap
from matplotlib import pyplot as plt
from scipy import stats

from research_pipeline.config import PipelineConfig
from research_pipeline.data_loading import collect_dates, collect_slice
from research_pipeline.feature_engineering import TARGET_COLUMN, feature_sets, required_columns
from research_pipeline.models import fit_predict, transformed_matrix


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
        train_start = dates[0]
        train_end = dates[cursor - 1]
        test_start = dates[cursor]
        test_end = dates[cursor + config.test_periods - 1]
        splits.append(
            RollingSplit(
                split_id=split_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        split_id += 1
        cursor += config.step_periods
    return splits


def directional_accuracy(actual: pd.Series, pred: pd.Series, current: pd.Series) -> float:
    actual_direction = np.sign(actual - current)
    pred_direction = np.sign(pred - current)
    return float((actual_direction == pred_direction).mean())


def metric_frame(model_name: str, estimator_name: str, segment: str, actual: pd.Series, pred: pd.Series, current: pd.Series) -> dict[str, object]:
    error = actual - pred
    denom = float(np.sum(np.square(actual - actual.mean())))
    r2 = float(1.0 - np.sum(np.square(error)) / denom) if denom > 0 else np.nan
    return {
        "model": model_name,
        "estimator": estimator_name,
        "segment": segment,
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "r2": r2,
        "directional_accuracy": directional_accuracy(actual, pred, current),
        "n_obs": int(len(actual)),
    }


def diebold_mariano(actual: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, horizon: int) -> dict[str, float]:
    loss_a = np.square(actual - pred_a)
    loss_b = np.square(actual - pred_b)
    diff = loss_a - loss_b
    n_obs = len(diff)
    if n_obs < 5:
        return {"dm_stat": np.nan, "p_value": np.nan}

    centered = diff - diff.mean()
    gamma0 = np.dot(centered, centered) / n_obs
    variance = gamma0
    for lag in range(1, max(horizon, 1)):
        weight = 1.0 - lag / horizon
        gamma = np.dot(centered[lag:], centered[:-lag]) / n_obs
        variance += 2.0 * weight * gamma
    variance = variance / n_obs
    if variance <= 0:
        return {"dm_stat": np.nan, "p_value": np.nan}

    dm_stat = diff.mean() / np.sqrt(variance)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value)}


def evaluate_models(lf: pl.LazyFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    dates = collect_dates(lf)
    splits = rolling_splits(dates, config)
    specs = feature_sets()
    estimators = ["linear", "random_forest", "xgboost"]
    columns = required_columns()

    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    artifacts: dict[str, object] = {}

    for split in splits:
        train_df = collect_slice(lf, split.train_start, split.train_end, columns)
        test_df = collect_slice(lf, split.test_start, split.test_end, columns)

        train_df = train_df.dropna(subset=[TARGET_COLUMN, "realized_volatility"])
        test_df = test_df.dropna(subset=[TARGET_COLUMN, "realized_volatility"])

        for model_name, feature_names in specs.items():
            for estimator_name in estimators:
                train_ready = train_df.dropna(subset=[TARGET_COLUMN, "asset"]).copy()
                test_ready = test_df.dropna(subset=[TARGET_COLUMN, "asset"]).copy()
                if train_ready.empty or test_ready.empty:
                    continue

                pipeline, prediction = fit_predict(train_ready, test_ready, feature_names, estimator_name, config)
                pred_df = test_ready[["date", "asset", "regime", "vix_regime", TARGET_COLUMN, "realized_volatility"]].copy()
                pred_df["prediction"] = prediction
                pred_df["model"] = model_name
                pred_df["estimator"] = estimator_name
                pred_df["split_id"] = split.split_id
                prediction_frames.append(pred_df)

                metric_rows.append(
                    metric_frame(
                        model_name,
                        estimator_name,
                        "overall",
                        pred_df[TARGET_COLUMN],
                        pred_df["prediction"],
                        pred_df["realized_volatility"],
                    )
                )
                for column in ["regime", "vix_regime"]:
                    for segment, segment_df in pred_df.groupby(column):
                        if segment_df.empty:
                            continue
                        metric_rows.append(
                            metric_frame(
                                model_name,
                                estimator_name,
                                f"{column}:{segment}",
                                segment_df[TARGET_COLUMN],
                                segment_df["prediction"],
                                segment_df["realized_volatility"],
                            )
                        )

                if model_name == "C_full" and estimator_name == "xgboost":
                    artifacts["shap_pipeline"] = pipeline
                    artifacts["shap_features"] = feature_names
                    artifacts["shap_frame"] = test_ready.copy()
                    artifacts["shap_split"] = split

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return metrics, predictions, artifacts


def aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["model", "estimator", "segment"], as_index=False)
        .agg(
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            r2=("r2", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
            n_obs=("n_obs", "sum"),
        )
    )


def dm_table(predictions: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for estimator_name, est_df in predictions.groupby("estimator"):
        pivot = est_df.pivot_table(
            index=["date", "asset", "split_id"],
            columns="model",
            values=["prediction", TARGET_COLUMN],
            aggfunc="first",
            observed=False,
        )
        pivot.columns = ["__".join(col) for col in pivot.columns]
        pivot = pivot.dropna().reset_index()
        actual = pivot[f"{TARGET_COLUMN}__A_base_macro"].to_numpy()
        for left, right in [("A_base_macro", "C_full"), ("B_base_macro_sentiment", "C_full")]:
            if f"prediction__{left}" not in pivot or f"prediction__{right}" not in pivot:
                continue
            result = diebold_mariano(
                actual,
                pivot[f"prediction__{left}"].to_numpy(),
                pivot[f"prediction__{right}"].to_numpy(),
                config.forecast_horizon,
            )
            rows.append(
                {
                    "estimator": estimator_name,
                    "comparison": f"{left} vs {right}",
                    **result,
                }
            )
    return pd.DataFrame(rows)


def regime_model_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    regime_predictions = predictions.loc[predictions["regime"].notna()].copy()
    for (segment, model_name, estimator_name), frame in regime_predictions.groupby(["regime", "model", "estimator"]):
        rows.append(
            metric_frame(
                model_name,
                estimator_name,
                segment,
                frame[TARGET_COLUMN],
                frame["prediction"],
                frame["realized_volatility"],
            )
        )
    return pd.DataFrame(rows)


def event_study_table(panel: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    macro_panel = (
        panel[["date", "poly_probability_change_lag_1", "target_forward_volatility"]]
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    signal = macro_panel["poly_probability_change_lag_1"]
    zscore = (signal - signal.mean()) / signal.std(ddof=0)
    event_dates = macro_panel.loc[zscore.abs() >= config.poly_spike_z, "date"]

    rows: list[dict[str, object]] = []
    for event_date in event_dates:
        anchor = macro_panel.index[macro_panel["date"] == event_date][0]
        for offset in range(-config.event_window, config.event_window + 1):
            idx = anchor + offset
            if idx < 0 or idx >= len(macro_panel):
                continue
            rows.append(
                {
                    "event_date": event_date,
                    "offset": offset,
                    "target_forward_volatility": macro_panel.iloc[idx]["target_forward_volatility"],
                }
            )
    event_df = pd.DataFrame(rows)
    if event_df.empty:
        return pd.DataFrame(columns=["offset", "avg_target_forward_volatility", "event_count"])
    return (
        event_df.groupby("offset", as_index=False)
        .agg(
            avg_target_forward_volatility=("target_forward_volatility", "mean"),
            event_count=("event_date", "nunique"),
        )
    )


def shap_outputs(artifacts: dict[str, object], config: PipelineConfig) -> pd.DataFrame:
    pipeline = artifacts["shap_pipeline"]
    feature_names = artifacts["shap_features"]
    test_frame = artifacts["shap_frame"].copy()
    sample = test_frame.sample(min(len(test_frame), config.shap_sample_size), random_state=config.random_state)
    matrix, transformed_names = transformed_matrix(pipeline, sample, feature_names)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()

    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer.shap_values(matrix)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({"feature": transformed_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, matrix, feature_names=transformed_names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(config.plot_dir / "shap_summary.png", dpi=config.plot_dpi, bbox_inches="tight")
    plt.close()

    target_feature = next((name for name in transformed_names if "poly_probability_level_lag_1" in name), None)
    if target_feature is not None:
        interaction_index = "auto"
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            target_feature,
            shap_values,
            matrix,
            feature_names=transformed_names,
            interaction_index=interaction_index,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(config.plot_dir / "shap_dependence_polymarket_probability.png", dpi=config.plot_dpi, bbox_inches="tight")
        plt.close()
    return importance


def save_plots(predictions: pd.DataFrame, metrics: pd.DataFrame, event_table: pd.DataFrame, config: PipelineConfig) -> None:
    best = (
        metrics.loc[metrics["segment"].eq("overall")]
        .sort_values(["rmse", "mae"])
        .iloc[0][["model", "estimator"]]
        .to_dict()
    )
    best_pred = predictions.loc[
        predictions["model"].eq(best["model"]) & predictions["estimator"].eq(best["estimator"])
    ].copy()
    plot_frame = best_pred.groupby("date", as_index=False).agg(
        actual=(TARGET_COLUMN, "mean"),
        predicted=("prediction", "mean"),
    )
    ax = plot_frame.plot(x="date", y=["actual", "predicted"], figsize=(11, 5), title="Predicted vs Actual Volatility")
    ax.set_ylabel("Volatility")
    plt.tight_layout()
    plt.savefig(config.plot_dir / "predicted_vs_actual_volatility.png", dpi=config.plot_dpi, bbox_inches="tight")
    plt.close()

    regime_plot = (
        metrics.loc[metrics["segment"].str.startswith("regime:")]
        .pivot_table(index="segment", columns=["model", "estimator"], values="rmse")
    )
    if not regime_plot.empty:
        regime_plot.plot(kind="bar", figsize=(12, 5), rot=0, title="Regime Comparison RMSE")
        plt.tight_layout()
        plt.savefig(config.plot_dir / "regime_comparison.png", dpi=config.plot_dpi, bbox_inches="tight")
        plt.close()

    if not event_table.empty:
        ax = event_table.plot(x="offset", y="avg_target_forward_volatility", marker="o", figsize=(8, 4), title="Event Study")
        ax.set_ylabel("Average Forward Volatility")
        plt.tight_layout()
        plt.savefig(config.plot_dir / "event_study.png", dpi=config.plot_dpi, bbox_inches="tight")
        plt.close()

    shap_plot = metrics.loc[metrics["segment"].eq("overall")].pivot(index="model", columns="estimator", values="rmse")
    if not shap_plot.empty:
        shap_plot.plot(kind="bar", figsize=(9, 5), rot=0, title="Model Comparison RMSE")
        plt.tight_layout()
        plt.savefig(config.plot_dir / "model_comparison_rmse.png", dpi=config.plot_dpi, bbox_inches="tight")
        plt.close()


def summary_text(metrics: pd.DataFrame, dm_results: pd.DataFrame, shap_importance: pd.DataFrame) -> str:
    overall = metrics.loc[metrics["segment"].eq("overall")].sort_values(["rmse", "mae"])
    best = overall.iloc[0]
    c_full = overall.loc[overall["model"].eq("C_full")].sort_values(["rmse", "mae"])
    c_best = c_full.iloc[0] if not c_full.empty else None
    base = overall.loc[
        overall["model"].eq("A_base_macro") & overall["estimator"].eq(c_best["estimator"] if c_best is not None else best["estimator"])
    ]
    sentiment_base = overall.loc[
        overall["model"].eq("B_base_macro_sentiment") & overall["estimator"].eq(c_best["estimator"] if c_best is not None else best["estimator"])
    ]
    base_rmse = float(base.iloc[0]["rmse"]) if not base.empty else np.nan
    sentiment_rmse = float(sentiment_base.iloc[0]["rmse"]) if not sentiment_base.empty else np.nan
    c_rmse = float(c_best["rmse"]) if c_best is not None else np.nan
    improve = bool(c_best is not None and c_rmse < min(base_rmse, sentiment_rmse))

    poly_rank = shap_importance.reset_index(drop=True)
    poly_rank["is_poly"] = poly_rank["feature"].str.contains("poly_", regex=False)
    poly_top = poly_rank.loc[poly_rank["is_poly"]].head(3)["feature"].tolist()

    dm_vs_a = dm_results.loc[dm_results["comparison"].eq("A_base_macro vs C_full")]
    dm_vs_b = dm_results.loc[dm_results["comparison"].eq("B_base_macro_sentiment vs C_full")]
    sig_vs_a = bool((dm_vs_a["p_value"] < 0.05).any())
    sig_vs_b = bool((dm_vs_b["p_value"] < 0.05).any())
    verdict = "YES" if improve else "NO"
    overall_best_uses_poly = bool(best["model"] == "C_full")
    evidence = (
        "Diebold-Mariano supports C_full against both simpler specifications."
        if sig_vs_a and sig_vs_b
        else "Diebold-Mariano does not support C_full as a robust improvement over both simpler specifications."
    )
    return (
        f"Do Polymarket features improve prediction? {verdict}\n"
        f"- Best model: {best['model']} with {best['estimator']}\n"
        f"- Best RMSE: {best['rmse']:.6f}\n"
        f"- Best C_full RMSE: {c_rmse:.6f}\n"
        f"- Matching A_base_macro RMSE: {base_rmse:.6f}\n"
        f"- Matching B_base_macro_sentiment RMSE: {sentiment_rmse:.6f}\n"
        f"- Is the single best overall model Polymarket-augmented? {'YES' if overall_best_uses_poly else 'NO'}\n"
        f"- Top Polymarket SHAP features: {', '.join(poly_top) if poly_top else 'none'}\n"
        f"- {evidence}\n"
    )
