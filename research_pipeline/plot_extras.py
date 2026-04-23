from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from research_pipeline.config import PipelineConfig


def _style() -> None:
    sns.set_theme(style="whitegrid", palette="deep")


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing table: {path}")
    return pd.read_csv(path)


def _save(fig: plt.Figure, path: Path, config: PipelineConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config.plot_dpi, bbox_inches="tight")
    plt.close(fig)


def plot_spike_improvement(config: PipelineConfig) -> None:
    _style()
    metrics = _load_table(config.table_dir / "spike_model_comparison.csv")
    overall = metrics.loc[metrics["segment"].eq("overall")].copy()

    left = overall.loc[overall["model"].eq("A_base_macro"), ["label_type", "estimator", "f1", "roc_auc"]].rename(
        columns={"f1": "f1_a", "roc_auc": "auc_a"}
    )
    right = overall.loc[overall["model"].eq("C_full"), ["label_type", "estimator", "f1", "roc_auc"]].rename(
        columns={"f1": "f1_c", "roc_auc": "auc_c"}
    )
    merged = left.merge(right, on=["label_type", "estimator"], how="inner")
    merged["f1_gain"] = merged["f1_c"] - merged["f1_a"]
    merged["auc_gain"] = merged["auc_c"] - merged["auc_a"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    f1_pivot = merged.pivot(index="estimator", columns="label_type", values="f1_gain")
    auc_pivot = merged.pivot(index="estimator", columns="label_type", values="auc_gain")
    sns.heatmap(f1_pivot, annot=True, fmt=".3f", center=0, cmap="RdYlGn", ax=axes[0])
    sns.heatmap(auc_pivot, annot=True, fmt=".3f", center=0, cmap="RdYlGn", ax=axes[1])
    axes[0].set_title("C_full minus A_base_macro: F1")
    axes[1].set_title("C_full minus A_base_macro: ROC-AUC")
    _save(fig, config.plot_dir / "spike_improvement_heatmaps.png", config)


def plot_bootstrap_forest(config: PipelineConfig) -> None:
    _style()
    bootstrap = _load_table(config.table_dir / "spike_bootstrap_results.csv")
    focus = bootstrap.loc[bootstrap["label_type"].eq("spike_top10")].copy()
    focus["label"] = focus["estimator"] + " | " + focus["comparison"] + " | " + focus["metric"]
    focus = focus.sort_values("diff_right_minus_left")

    fig, ax = plt.subplots(figsize=(11, 8))
    y = range(len(focus))
    lower = focus["diff_right_minus_left"] - focus["ci_low"]
    upper = focus["ci_high"] - focus["diff_right_minus_left"]
    ax.errorbar(
        focus["diff_right_minus_left"],
        y,
        xerr=[lower, upper],
        fmt="o",
        color="#1f77b4",
        ecolor="#4c78a8",
        capsize=3,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(focus["label"])
    ax.set_title("Bootstrap Confidence Intervals for Spike Improvement")
    ax.set_xlabel("C_full minus benchmark")
    _save(fig, config.plot_dir / "spike_bootstrap_forest.png", config)


def plot_regime_bars(config: PipelineConfig) -> None:
    _style()
    regime = _load_table(config.table_dir / "spike_regime_comparison.csv")
    focus = regime.loc[
        regime["label_type"].eq("spike_top10") & regime["segment"].isin(["high_vix", "low_vix"])
    ].copy()
    focus["model_estimator"] = focus["model"] + "\n" + focus["estimator"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, segment in zip(axes, ["high_vix", "low_vix"]):
        frame = focus.loc[focus["segment"].eq(segment)].sort_values("f1", ascending=False)
        sns.barplot(data=frame, x="model_estimator", y="f1", ax=ax, color="#4c78a8")
        ax.set_title(f"Spike Top10 F1 in {segment}")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("F1")
    axes[1].set_ylabel("")
    _save(fig, config.plot_dir / "spike_regime_f1_bars.png", config)


def plot_shap_views(config: PipelineConfig) -> None:
    _style()
    shap_df = _load_table(config.table_dir / "spike_shap_importance.csv")
    top = shap_df.head(15).copy().sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4c78a8")
    ax.set_title("Top SHAP Features for Spike Classification")
    ax.set_xlabel("Mean |SHAP|")
    _save(fig, config.plot_dir / "spike_shap_top15.png", config)

    poly = shap_df.loc[shap_df["feature"].str.contains("poly_", regex=False)].head(12).copy()
    if not poly.empty:
        poly = poly.sort_values("mean_abs_shap")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(poly["feature"], poly["mean_abs_shap"], color="#f58518")
        ax.set_title("Polymarket SHAP Features")
        ax.set_xlabel("Mean |SHAP|")
        _save(fig, config.plot_dir / "spike_shap_poly_only.png", config)


def plot_model_rankings(config: PipelineConfig) -> None:
    _style()
    metrics = _load_table(config.table_dir / "spike_model_comparison.csv")
    focus = metrics.loc[metrics["segment"].eq("overall") & metrics["label_type"].eq("spike_top10")].copy()
    focus["model_estimator"] = focus["model"] + "\n" + focus["estimator"]
    focus = focus.sort_values("f1", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=focus, x="model_estimator", y="f1", ax=axes[0], color="#54a24b")
    sns.barplot(data=focus, x="model_estimator", y="roc_auc", ax=axes[1], color="#e45756")
    axes[0].set_title("Overall Spike Top10 F1")
    axes[1].set_title("Overall Spike Top10 ROC-AUC")
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("F1")
    axes[1].set_ylabel("ROC-AUC")
    _save(fig, config.plot_dir / "spike_model_rankings.png", config)


def generate_extra_plots(config: PipelineConfig | None = None) -> None:
    config = config or PipelineConfig()
    plot_spike_improvement(config)
    plot_bootstrap_forest(config)
    plot_regime_bars(config)
    plot_shap_views(config)
    plot_model_rankings(config)
