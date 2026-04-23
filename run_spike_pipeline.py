from research_pipeline import PipelineConfig, run_spike_pipeline


def main() -> None:
    result = run_spike_pipeline(PipelineConfig())
    metrics = result["metrics"].copy()
    overall = metrics.loc[
        metrics["segment"].eq("overall"),
        ["label_type", "model", "estimator", "precision", "recall", "f1", "roc_auc"],
    ]
    bootstrap = result["bootstrap"].copy()

    print("\nSpike Classification Metrics")
    print(overall.to_string(index=False))
    print("\nBootstrap Comparison")
    print(bootstrap.to_string(index=False))
    print("\nFinal Summary")
    print(result["summary"])


if __name__ == "__main__":
    main()
