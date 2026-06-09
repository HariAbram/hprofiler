"""Abstract base for LLM API providers."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(ABC):
    def __init__(self, model: str, api_key: str | None = None, endpoint: str | None = None) -> None:
        self.model = model
        self.api_key = api_key
        self.endpoint = endpoint

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.2,
    ) -> ChatResponse:
        """Send messages and return response.

        messages: OpenAI-format list:
          {"role": "user"|"assistant", "content": str}
          {"role": "assistant", "content": str|None, "tool_calls": [{"id", "name", "arguments"}]}
          {"role": "tool", "tool_call_id": str, "content": str}

        tools: OpenAI function-calling format tool list.
        system: system prompt (separate param; prepended as system message for OpenAI).
        """

    @property
    def display_name(self) -> str:
        return f"{type(self).__name__.replace('Provider', '')} / {self.model}"

    def _http_post(self, url: str, headers: dict, body: dict, timeout: int = 600) -> dict:
        import json
        import urllib.request
        import urllib.error

        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {e.code} from {url}: {raw[:600]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Cannot reach {url}: {e.reason}") from e
