from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AgentNote:
    role: str
    thesis: str
    signal_strength: str
    confidence: float
    key_points: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    watch_items: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], role: str) -> "AgentNote":
        return cls(
            role=role,
            thesis=str(payload.get("thesis", "")).strip(),
            signal_strength=str(payload.get("signal_strength", "medium")).strip(),
            confidence=float(payload.get("confidence", 0.5) or 0.5),
            key_points=[str(item).strip() for item in payload.get("key_points", []) if str(item).strip()],
            risk_flags=[str(item).strip() for item in payload.get("risk_flags", []) if str(item).strip()],
            watch_items=[str(item).strip() for item in payload.get("watch_items", []) if str(item).strip()],
        )


@dataclass
class ResearchVerdict:
    asset: str
    as_of_date: str
    stance: str
    horizon: str
    conviction: float
    summary: str
    supporting_points: list[str] = field(default_factory=list)
    invalidation_triggers: list[str] = field(default_factory=list)
    follow_up_tests: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], asset: str, as_of_date: str) -> "ResearchVerdict":
        return cls(
            asset=asset,
            as_of_date=as_of_date,
            stance=str(payload.get("stance", "neutral")).strip(),
            horizon=str(payload.get("horizon", "1-5 trading days")).strip(),
            conviction=float(payload.get("conviction", 0.5) or 0.5),
            summary=str(payload.get("summary", "")).strip(),
            supporting_points=[
                str(item).strip() for item in payload.get("supporting_points", []) if str(item).strip()
            ],
            invalidation_triggers=[
                str(item).strip() for item in payload.get("invalidation_triggers", []) if str(item).strip()
            ],
            follow_up_tests=[
                str(item).strip() for item in payload.get("follow_up_tests", []) if str(item).strip()
            ],
        )


def as_payload(value: AgentNote | ResearchVerdict) -> dict[str, Any]:
    return asdict(value)
