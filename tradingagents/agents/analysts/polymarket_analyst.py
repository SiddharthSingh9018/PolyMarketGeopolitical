from langchain_core.messages import AIMessage

from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction
from tradingagents.agents.utils.polymarket_context import build_polymarket_context


def create_polymarket_analyst(llm):
    def polymarket_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = build_instrument_context(ticker)
        poly_context = build_polymarket_context(ticker, current_date)

        prompt = "\n\n".join(
            [
                "You are the Polymarket Analyst for a research-focused geopolitical trading workflow.",
                "Use the supplied Polymarket-derived panel context to explain whether prediction-market activity adds actionable short-horizon insight.",
                "You cannot call external tools in this step. Stay grounded in the supplied context only.",
                "Write a concise but specific markdown report with these sections:",
                "1. Signal Summary",
                "2. Event/Regime Interpretation",
                "3. Why Polymarket Matters or Does Not Matter Here",
                "4. What To Watch Next",
                "Explicitly say when the Polymarket signal is absent, weak, or redundant relative to macro stress.",
                get_language_instruction(),
                f"Current date: {current_date}",
                instrument_context,
                poly_context,
            ]
        )
        result = llm.invoke(prompt)
        report = result.content if hasattr(result, "content") else str(result)
        message = AIMessage(content=report, name="Polymarket Analyst")
        return {
            "messages": [message],
            "polymarket_report": report,
            "sender": "Polymarket Analyst",
        }

    return polymarket_analyst_node
