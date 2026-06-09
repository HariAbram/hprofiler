"""Anthropic Messages API provider (claude-* models)."""

from __future__ import annotations
from .base import LLMProvider, ChatResponse, ToolCall

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"
_DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicProvider(LLMProvider):

    def chat(self, messages, system="", tools=None, max_tokens=4096, temperature=0.2):
        if not self.api_key:
            raise RuntimeError(
                "Anthropic API key required. "
                "Set ANTHROPIC_API_KEY or pass --llm-api-key."
            )

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }

        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": self._convert_messages(messages),
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = self._convert_tools(tools)

        resp = self._http_post(_API_URL, headers, body)

        text = ""
        tool_calls: list[ToolCall] = []
        for block in resp.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(ToolCall(
                    id=block["id"],
                    name=block["name"],
                    arguments=block.get("input", {}),
                ))

        return ChatResponse(content=text.strip(), tool_calls=tool_calls)

    def _convert_messages(self, messages: list[dict]) -> list[dict]:
        """Translate from OpenAI internal format to Anthropic format."""
        result: list[dict] = []
        for m in messages:
            role = m["role"]
            if role == "user":
                result.append({"role": "user", "content": m["content"]})
            elif role == "assistant":
                content: list = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls", []):
                    content.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["arguments"],
                    })
                if not content:
                    content = [{"type": "text", "text": ""}]
                result.append({"role": "assistant", "content": content})
            elif role == "tool":
                # Batch consecutive tool results into a single user message
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m["content"],
                }
                if result and result[-1]["role"] == "user" and isinstance(result[-1]["content"], list):
                    result[-1]["content"].append(tool_result)
                else:
                    result.append({"role": "user", "content": [tool_result]})
        return result

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Translate from OpenAI function-calling format to Anthropic tool format."""
        result = []
        for t in tools:
            fn = t.get("function", t)
            result.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return result
