from research_pipeline import PipelineConfig, run_large_scale_pipeline


def main() -> None:
    result = run_large_scale_pipeline(PipelineConfig())
    metrics = result["metrics"].copy()
    overall = metrics.loc[metrics["segment"].eq("overall"), ["model", "estimator", "rmse", "mae", "directional_accuracy"]]
    dm = result["dm_results"].copy()

    print("\nOverall Evaluation")
    print(overall.to_string(index=False))
    print("\nDiebold-Mariano")
    print(dm.to_string(index=False))
    print("\nFinal Summary")
    print(result["summary"])


if __name__ == "__main__":
    main()
