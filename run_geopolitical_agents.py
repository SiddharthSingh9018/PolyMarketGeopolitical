from __future__ import annotations

import argparse
from pathlib import Path

from geopolitical_agents import AgentConfig, GeopoliticalResearchOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the research-only geopolitical roundtable prototype.")
    parser.add_argument("--asset", required=True, help="Ticker present in model_dataset.csv")
    parser.add_argument("--date", required=True, help="As-of date in YYYY-MM-DD format")
    parser.add_argument("--provider", default="groq", choices=["groq", "local_openai"])
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--source-note", default=None, help="Optional markdown source note to inject as context")
    parser.add_argument("--inter-agent-delay", type=float, default=1.5)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--lookback-rows", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AgentConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        inter_agent_delay_seconds=args.inter_agent_delay,
        max_tokens=args.max_tokens,
        lookback_rows=args.lookback_rows,
    )
    orchestrator = GeopoliticalResearchOrchestrator(config)
    artifact = orchestrator.run(
        asset=args.asset,
        as_of_date=args.date,
        source_note_path=Path(args.source_note) if args.source_note else None,
    )
    verdict = artifact["verdict"]
    print(f"Asset: {verdict['asset']}")
    print(f"As of: {verdict['as_of_date']}")
    print(f"Stance: {verdict['stance']}")
    print(f"Conviction: {verdict['conviction']}")
    print(f"Summary: {verdict['summary']}")


if __name__ == "__main__":
    main()
