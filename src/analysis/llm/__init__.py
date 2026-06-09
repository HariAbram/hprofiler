"""LLM provider factory and auto-detection."""

from __future__ import annotations
import os
from .base import LLMProvider


def _ollama_reachable(base_url: str) -> bool:
    import urllib.request
    try:
        url = base_url.rstrip("/").removesuffix("/v1") + "/api/tags"
        with urllib.request.urlopen(url, timeout=2):
            return True
    except Exception:
        return False


def auto_detect() -> tuple[str, str, str | None, str | None]:
    """Probe environment for an available LLM provider.

    Returns (provider, model, api_key, endpoint).
    provider is empty string when nothing is detected.
    """
    # Anthropic
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if key:
        model = os.getenv("HPROFILER_LLM_MODEL", "claude-sonnet-4-6")
        return "anthropic", model, key, None

    # OpenAI
    key = os.getenv("OPENAI_API_KEY", "")
    if key:
        model = os.getenv("HPROFILER_LLM_MODEL", "gpt-4o")
        return "openai", model, key, None

    # Ollama (local)
    ollama_base = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if _ollama_reachable(ollama_base):
        model = os.getenv("HPROFILER_LLM_MODEL", "llama3.1:8b")
        return "ollama", model, None, ollama_base

    return "", "", None, None


def create_provider(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    endpoint: str | None = None,
) -> LLMProvider:
    """Create an LLM provider instance.

    provider: "anthropic" | "openai" | "ollama" | "openai-compat" | None (auto)
    model:    any model string the chosen provider accepts
              (e.g. "llama3.1:8b", "gpt-4o", "claude-opus-4-8")
    api_key:  API key; falls back to ANTHROPIC_API_KEY / OPENAI_API_KEY env vars
    endpoint: base URL for openai-compat / ollama
              (e.g. "http://localhost:11434" or "https://api.groq.com/openai/v1")

    Environment overrides (lower priority than explicit args):
      HPROFILER_LLM_PROVIDER, HPROFILER_LLM_MODEL,
      HPROFILER_LLM_API_KEY,  HPROFILER_LLM_ENDPOINT
    """
    # Fill gaps from env vars
    provider  = provider  or os.getenv("HPROFILER_LLM_PROVIDER") or None
    model     = model     or os.getenv("HPROFILER_LLM_MODEL")     or None
    api_key   = api_key   or os.getenv("HPROFILER_LLM_API_KEY")   or None
    endpoint  = endpoint  or os.getenv("HPROFILER_LLM_ENDPOINT")  or None

    # Auto-detect if provider still unknown
    if not provider:
        det_prov, det_model, det_key, det_ep = auto_detect()
        if not det_prov:
            raise RuntimeError(
                "No LLM provider found. Options:\n"
                "  • Set ANTHROPIC_API_KEY to use Claude\n"
                "  • Set OPENAI_API_KEY to use GPT\n"
                "  • Run 'ollama serve' for a local model\n"
                "  • Pass --llm anthropic|openai|ollama|openai-compat"
            )
        provider = det_prov
        model    = model    or det_model
        api_key  = api_key  or det_key
        endpoint = endpoint or det_ep

    # Fill API key from standard env vars if still absent
    if not api_key:
        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY") or None
        elif provider in ("openai", "openai-compat"):
            api_key = os.getenv("OPENAI_API_KEY") or None

    from .anthropic import AnthropicProvider
    from .openai_compat import OpenAICompatProvider

    if provider == "anthropic":
        return AnthropicProvider(model or "claude-sonnet-4-6", api_key=api_key)

    if provider == "openai":
        return OpenAICompatProvider(
            model or "gpt-4o",
            api_key=api_key,
            endpoint=endpoint or "https://api.openai.com/v1",
        )

    if provider == "ollama":
        base = (endpoint or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        return OpenAICompatProvider(
            model or "llama3.1:8b",
            api_key="ollama",
            endpoint=base,
        )

    if provider == "openai-compat":
        if not endpoint:
            raise RuntimeError(
                "--llm openai-compat requires --llm-endpoint <base-url>\n"
                "Example: --llm-endpoint http://localhost:8080"
            )
        return OpenAICompatProvider(model or "default", api_key=api_key, endpoint=endpoint)

    raise ValueError(
        f"Unknown provider '{provider}'. "
        "Valid: anthropic, openai, ollama, openai-compat"
    )
