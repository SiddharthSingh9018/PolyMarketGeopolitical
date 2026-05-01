# Geopolitical Agents Prototype

This prototype adds a research-only Tauric-inspired roundtable on top of the existing Polymarket research dataset.

## Purpose

The current `poly_data` pipeline is strongest as:

- a data-ingestion layer
- a feature-engineering layer
- a forecasting and spike-detection research layer

It is weaker as:

- a decision-orchestration system
- an event-thesis generator
- a reusable analyst workflow

The new prototype addresses that gap.

## Design

The scaffold uses a compact multi-agent sequence inspired by `TradingAgents`, but tailored to your research constraints:

- `polymarket_analyst`
- `macro_analyst`
- `roundtable_manager`

The `roundtable_manager` emits:

- a `bull_researcher` case
- a `bear_researcher` case
- a final `research_manager`-style verdict

Unlike Tauric's full system, this prototype:

- is research-only
- does not submit or simulate orders
- uses the existing `model_dataset.csv` panel as the core context source
- supports only Groq or a local OpenAI-compatible endpoint

## Inputs

For a chosen `asset` and `date`, the prototype reads the latest available row in:

- `research_data/processed/model_dataset.csv`

and builds context from:

- market state
- Polymarket features
- macro conditions
- sentiment
- recent lookback rows
- derived event flags such as `poly_jump` and `high_vix`

An optional external source note can also be injected, which makes it easy to combine:

- clipped GitHub research notes
- article summaries
- policy or geopolitical notes

with the quantitative context.

## Why this is a better next step

Your empirical results suggest:

- broad daily forecasting gains from Polymarket are weak
- event-sensitive value is more plausible
- high-stress regimes are more promising than unconditional forecasting

That makes a Tauric-style research roundtable a better fit than another purely tabular benchmark. The multi-agent layer lets Polymarket operate as:

- an event-risk signal
- a debate input
- a conditional geopolitical trigger

rather than an always-on universal predictor.

## Run command

```bash
uv run python run_geopolitical_agents.py --asset LMT --date 2020-06-15
```

Optional source-note injection:

```bash
uv run python run_geopolitical_agents.py --asset RTX --date 2020-06-15 --source-note "C:/path/to/source.md"
```

## Output

Artifacts are written to:

- `research_outputs/geopolitical_agents/`

Each run produces:

- one `.json` artifact
- one `.md` roundtable artifact

## Next recommended research tasks

1. Add curated Polymarket market-selection logic for geopolitical themes rather than broad keyword matching.
2. Compare roundtable quality with and without a clipped external source note.
3. Restrict experiments to high-VIX or event-driven windows.
4. Add defense-specific and broader geopolitical asset baskets as separate evaluation tracks.
