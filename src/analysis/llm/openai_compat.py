"""OpenAI chat completions API and any compatible endpoint.

Works with OpenAI, Ollama (/v1), vLLM, LM Studio, Together.ai, Groq, etc.
"""

from __future__ import annotations
import json
from .base import LLMProvider, ChatResponse, ToolCall

_OPENAI_BASE = "https://api.openai.com/v1"


class OpenAICompatProvider(LLMProvider):

    def __init__(self, model: str, api_key: str | None = None, endpoint: str | None = None) -> None:
        base = (endpoint or _OPENAI_BASE).rstrip("/")
        # Normalise: append /chat/completions if endpoint is a base URL
        if not base.endswith("/chat/completions"):
            if not base.endswith("/v1"):
                base = base + "/v1"
            base = base + "/chat/completions"
        super().__init__(model, api_key=api_key, endpoint=base)

    def chat(self, messages, system="", tools=None, max_tokens=4096, temperature=0.2):
        headers: dict = {"content-type": "application/json"}
        if self.api_key and self.api_key != "ollama":
            headers["Authorization"] = f"Bearer {self.api_key}"

        all_messages: list[dict] = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(self._normalize(messages))

        body: dict = {
            "model": self.model,
            "messages": all_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        resp = self._http_post(self.endpoint, headers, body)  # type: ignore[arg-type]

        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or ""
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append(ToolCall(id=tc["id"], name=fn.get("name", ""), arguments=args))

        return ChatResponse(content=content.strip(), tool_calls=tool_calls)

    def _normalize(self, messages: list[dict]) -> list[dict]:
        """Ensure messages conform to the OpenAI wire format."""
        result = []
        for m in messages:
            role = m["role"]
            if role == "user":
                result.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in m["tool_calls"]
                    ]
                result.append(msg)
            elif role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": m["tool_call_id"],
                    "content": m["content"],
                })
        return result
