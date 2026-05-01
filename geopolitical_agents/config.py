from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AgentConfig:
    dataset_path: Path = Path("research_data/processed/model_dataset.csv")
    output_dir: Path = Path("research_outputs/geopolitical_agents")
    provider: str = "groq"
    model: str = "openai/gpt-oss-20b"
    base_url: Optional[str] = None
    api_key_env: str = "GROQ_API_KEY"
    local_api_key_env: str = "LOCAL_LLM_API_KEY"
    local_base_url_env: str = "LOCAL_LLM_BASE_URL"
    temperature: float = 0.2
    max_tokens: int = 900
    lookback_rows: int = 15
    include_source_note: bool = True
    source_note_char_limit: int = 3500
    request_retries: int = 3
    retry_backoff_seconds: float = 2.0
    inter_agent_delay_seconds: float = 1.5

    def resolved_output_dir(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir
