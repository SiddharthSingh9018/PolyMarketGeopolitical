from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests

from geopolitical_agents.config import AgentConfig


def _extract_json_blob(text: str) -> dict[str, Any]:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return json.loads(text[start : end + 1])


@dataclass
class OpenAICompatibleClient:
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    request_retries: int
    retry_backoff_seconds: float

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = None
        for attempt in range(self.request_retries):
            response = requests.post(
                f"{self.base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=180,
            )
            if response.ok:
                break
            if response.status_code != 429 or attempt == self.request_retries - 1:
                response.raise_for_status()
            time.sleep(self.retry_backoff_seconds * (attempt + 1))

        if response is None:
            raise RuntimeError("No response received from model provider.")

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return _extract_json_blob(content)


def build_client(config: AgentConfig) -> OpenAICompatibleClient:
    provider = config.provider.lower()
    if provider == "groq":
        api_key = os.getenv(config.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing {config.api_key_env} for Groq provider.")
        base_url = config.base_url or "https://api.groq.com/openai/v1"
        return OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            request_retries=config.request_retries,
            retry_backoff_seconds=config.retry_backoff_seconds,
        )

    if provider == "local_openai":
        base_url = config.base_url or os.getenv(config.local_base_url_env, "http://127.0.0.1:11434/v1")
        api_key = os.getenv(config.local_api_key_env, "local")
        return OpenAICompatibleClient(
            base_url=base_url,
            api_key=api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            request_retries=config.request_retries,
            retry_backoff_seconds=config.retry_backoff_seconds,
        )

    raise ValueError(f"Unsupported provider: {config.provider}")
