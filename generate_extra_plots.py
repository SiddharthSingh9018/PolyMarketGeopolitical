from research_pipeline import PipelineConfig
from research_pipeline.plot_extras import generate_extra_plots


def main() -> None:
    generate_extra_plots(PipelineConfig())
    print("Extra plots saved to research_outputs/plots")


if __name__ == "__main__":
    main()
