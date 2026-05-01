from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from geopolitical_agents.clients import build_client
from geopolitical_agents.config import AgentConfig
from geopolitical_agents.context import ResearchContext, build_research_context
from geopolitical_agents.schemas import AgentNote, ResearchVerdict, as_payload


SYSTEM_PROMPT = """You are part of a research-only multi-agent geopolitical trading prototype.
Use only the supplied context.
Do not claim live market access.
Respond with valid JSON only.
Keep the reasoning grounded, concise, and falsifiable."""


def _json_contract(role: str) -> str:
    if role == "roundtable_manager":
        return """Return JSON with keys:
{
  "bull_case": "2-4 sentence bullish case",
  "bear_case": "2-4 sentence bearish case",
  "verdict": {
    "stance": "long|neutral|short",
    "horizon": "string",
    "conviction": 0.0,
    "summary": "string",
    "supporting_points": ["..."],
    "invalidation_triggers": ["..."],
    "follow_up_tests": ["..."]
  }
}"""
    if role == "research_manager":
        return """Return JSON with keys:
{
  "stance": "long|neutral|short",
  "horizon": "string",
  "conviction": 0.0,
  "summary": "string",
  "supporting_points": ["..."],
  "invalidation_triggers": ["..."],
  "follow_up_tests": ["..."]
}"""
    return """Return JSON with keys:
{
  "thesis": "string",
  "signal_strength": "low|medium|high",
  "confidence": 0.0,
  "key_points": ["..."],
  "risk_flags": ["..."],
  "watch_items": ["..."]
}"""


def _role_prompt(role: str, context: ResearchContext, prior_notes: Optional[List[AgentNote]] = None) -> str:
    notes_block = ""
    if prior_notes:
        rendered = []
        for note in prior_notes:
            rendered.append(
                "\n".join(
                    [
                        f"Role: {note.role}",
                        f"Thesis: {note.thesis}",
                        f"Signal strength: {note.signal_strength}",
                        f"Confidence: {note.confidence}",
                        f"Key points: {'; '.join(note.key_points)}",
                        f"Risk flags: {'; '.join(note.risk_flags)}",
                        f"Watch items: {'; '.join(note.watch_items)}",
                    ]
                )
            )
        notes_block = "\n\nPrior notes:\n" + "\n\n".join(rendered)

    role_instructions = {
        "polymarket_analyst": (
            "Focus on Polymarket probability, probability change, volume anomalies, order imbalance, "
            "and whether the current setup looks event-driven or regime-driven."
        ),
        "macro_analyst": (
            "Focus on VIX, oil-volatility proxy, WTI, GPR, and sentiment to judge broader geopolitical stress."
        ),
        "bull_researcher": (
            "Make the strongest credible bullish case for the asset using the analyst notes and context. "
            "Be specific about why the geopolitical setup could help the asset."
        ),
        "bear_researcher": (
            "Make the strongest credible bearish case for the asset using the analyst notes and context. "
            "Be specific about why the geopolitical setup could hurt the asset or fail to matter."
        ),
        "roundtable_manager": (
            "Produce a compact roundtable result containing a bullish case, a bearish case, and a final research verdict. "
            "This should feel like Tauric-style debate compressed into one structured synthesis call for research efficiency."
        ),
        "research_manager": (
            "Synthesize the analyst and debate notes into a research verdict for a short-horizon geopolitical trading thesis. "
            "This is research only, not execution advice."
        ),
    }

    return "\n\n".join(
        [
            f"Role: {role}",
            role_instructions[role],
            _json_contract(role),
            "Context:",
            context.to_markdown(),
            notes_block,
        ]
    )


class GeopoliticalResearchOrchestrator:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = build_client(config)

    def _run_note(
        self,
        role: str,
        context: ResearchContext,
        prior_notes: Optional[List[AgentNote]] = None,
    ) -> AgentNote:
        payload = self.client.chat_json(SYSTEM_PROMPT, _role_prompt(role, context, prior_notes))
        if self.config.inter_agent_delay_seconds > 0:
            time.sleep(self.config.inter_agent_delay_seconds)
        return AgentNote.from_dict(payload, role=role)

    def _run_roundtable(
        self,
        context: ResearchContext,
        prior_notes: List[AgentNote],
    ) -> Tuple[AgentNote, AgentNote, ResearchVerdict]:
        payload = self.client.chat_json(SYSTEM_PROMPT, _role_prompt("roundtable_manager", context, prior_notes))
        if self.config.inter_agent_delay_seconds > 0:
            time.sleep(self.config.inter_agent_delay_seconds)
        bull_note = AgentNote(
            role="bull_researcher",
            thesis=str(payload.get("bull_case", "")).strip(),
            signal_strength="medium",
            confidence=0.5,
            key_points=[],
            risk_flags=[],
            watch_items=[],
        )
        bear_note = AgentNote(
            role="bear_researcher",
            thesis=str(payload.get("bear_case", "")).strip(),
            signal_strength="medium",
            confidence=0.5,
            key_points=[],
            risk_flags=[],
            watch_items=[],
        )
        verdict = ResearchVerdict.from_dict(
            payload.get("verdict", {}),
            asset=context.asset,
            as_of_date=context.as_of_date,
        )
        return bull_note, bear_note, verdict

    def run(self, asset: str, as_of_date: str, source_note_path: Optional[Path] = None) -> dict[str, Any]:
        context = build_research_context(self.config, asset=asset, as_of_date=as_of_date, source_note_path=source_note_path)
        poly_note = self._run_note("polymarket_analyst", context)
        macro_note = self._run_note("macro_analyst", context)
        bull_note, bear_note, verdict = self._run_roundtable(context, [poly_note, macro_note])
        artifact = {
            "context": {
                "asset": context.asset,
                "as_of_date": context.as_of_date,
                "snapshot": context.snapshot,
                "derived_flags": context.derived_flags,
                "recent_rows": context.recent_rows,
                "source_note_included": bool(context.source_note),
            },
            "notes": [as_payload(note) for note in [poly_note, macro_note, bull_note, bear_note]],
            "verdict": as_payload(verdict),
        }
        self._save_artifacts(asset, context.as_of_date, artifact)
        return artifact

    def _save_artifacts(self, asset: str, as_of_date: str, artifact: dict[str, Any]) -> None:
        output_dir = self.config.resolved_output_dir()
        stem = f"{as_of_date}_{asset}_geopolitical_roundtable"
        json_path = output_dir / f"{stem}.json"
        md_path = output_dir / f"{stem}.md"
        json_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")

        verdict = artifact["verdict"]
        lines = [
            f"# Geopolitical Research Roundtable: {asset}",
            "",
            f"- As of date: {artifact['context']['as_of_date']}",
            f"- Stance: {verdict['stance']}",
            f"- Horizon: {verdict['horizon']}",
            f"- Conviction: {verdict['conviction']}",
            "",
            "## Summary",
            verdict["summary"],
            "",
            "## Supporting Points",
        ]
        lines.extend(f"- {item}" for item in verdict["supporting_points"])
        lines.extend(["", "## Invalidation Triggers"])
        lines.extend(f"- {item}" for item in verdict["invalidation_triggers"])
        lines.extend(["", "## Follow-up Tests"])
        lines.extend(f"- {item}" for item in verdict["follow_up_tests"])
        lines.extend(["", "## Agent Notes"])
        for note in artifact["notes"]:
            lines.extend(
                [
                    "",
                    f"### {note['role']}",
                    f"- Signal strength: {note['signal_strength']}",
                    f"- Confidence: {note['confidence']}",
                    f"- Thesis: {note['thesis']}",
                    "- Key points:",
                ]
            )
            lines.extend(f"  - {item}" for item in note["key_points"])
            lines.append("- Risk flags:")
            lines.extend(f"  - {item}" for item in note["risk_flags"])
            lines.append("- Watch items:")
            lines.extend(f"  - {item}" for item in note["watch_items"])
        md_path.write_text("\n".join(lines), encoding="utf-8")
