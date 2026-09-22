"""
llm_provider.py — Centralised LLM selection for the AI Agent Engineer course.

Every demo, solution, starter, and project in this course imports from this
module instead of instantiating ``ChatOpenAI`` directly. That single indirection
gives the whole course a working **provider switch** for free.

Two providers are supported today:

    1. ``openai``  (default) — OpenAI's API. Drop-in, fully backward-compatible
       with anything you see in the lectures. Uses ``OPENAI_API_KEY`` and
       ``OPENAI_MODEL``.

    2. ``minimax`` — MiniMax M3 (and other MiniMax models) reached via the
       MiniMax OpenAI-compatible endpoint. No extra dependencies; we still
       use ``ChatOpenAI`` under the hood, just pointed at a different
       ``base_url``. Set ``MINIMAX_API_KEY``.

Switching providers is an env-var flip — no code change required:

    # Default — uses OpenAI, identical to before this module existed
    python 04-01-minimal-agent.py

    # Switch to MiniMax M3
    LLM_PROVIDER=minimax MINIMAX_API_KEY=sk-... python 04-01-minimal-agent.py

Usage in code
-------------

    from llm_provider import get_chat_model, get_provider_info

    llm = get_chat_model()                       # default model + temperature
    llm = get_chat_model(model="gpt-4o")         # one-off override
    llm = get_chat_model(temperature=0.3)        # one-off override

    info = get_provider_info()                   # {'provider': 'minimax',
                                                #  'model': 'MiniMax-M3',
                                                #  'base_url': '...'}
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# Load the course-level .env regardless of the directory a demo is run from.
# Existing shell environment variables keep priority over values in the file.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def clean_model_text(content: Any) -> str:
    """Return user-facing model text without provider reasoning blocks.

    Some reasoning models include internal reasoning inside ``<think>`` tags.
    Other models return plain text, which passes through unchanged.
    """
    if content is None:
        return ""

    text = content if isinstance(content, str) else str(content)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


# ── Provider presets ───────────────────────────────────────────────────────────
# Tweak defaults here once — every demo, solution, and project picks them up.

OPENAI_DEFAULTS = {
    "model": "gpt-4o-mini",
    "temperature": 0,
}

MINIMAX_DEFAULTS = {
    "model": "MiniMax-M3",
    "base_url": "https://api.minimax.io/v1",
    "temperature": 0,
}


def _resolve_provider() -> str:
    """Return the active provider name. Defaults to ``'openai'``."""
    return (os.getenv("LLM_PROVIDER") or "openai").strip().lower()


def _env_float(key: str, default: float) -> float:
    """Read a float from env, falling back to ``default`` on any error."""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    """Read a bool-ish env var, accepting common true/false spellings."""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _log_provider_use(**fields: Any) -> None:
    """Emit a compact provider trace line unless disabled by env."""
    if not _env_bool("LLM_PROVIDER_OBSERVABILITY", True):
        return

    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(f"[llm_provider] {details}", file=sys.stderr)


def get_chat_model(
    model: str | None = None,
    temperature: float | None = None,
    **overrides: Any,
):
    """Return a chat model for the active provider.

    Args:
        model: Override the model name. If ``None``, uses the provider default
            (or the provider-specific env var, e.g. ``OPENAI_MODEL`` /
            ``MINIMAX_MODEL``).
        temperature: Override the temperature. If ``None``, reads the
            ``TEMPERATURE`` env var, falling back to the provider default.
        **overrides: Extra kwargs forwarded to the underlying ``ChatOpenAI``
            constructor (``timeout``, ``max_tokens``, ``max_retries`` …).

    Returns:
        A LangChain ``BaseChatModel`` — typically a ``ChatOpenAI`` instance.

    Raises:
        RuntimeError: If the active provider is missing its API key.
        ValueError: If ``LLM_PROVIDER`` is set to an unknown value.
    """
    provider = _resolve_provider()

    if provider == "minimax":
        return _build_minimax(model, temperature, **overrides)
    if provider == "openai":
        return _build_openai(model, temperature, **overrides)

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Supported values: 'openai' (default), "
        f"'minimax'."
    )


def _build_openai(model, temperature, **overrides):
    """Build a standard OpenAI chat model. Identical to the pre-existing
    ``ChatOpenAI(model="gpt-4o-mini", temperature=0)`` pattern used throughout
    the course — no surprises."""
    from langchain_openai import ChatOpenAI

    resolved_model = model or os.getenv("OPENAI_MODEL", OPENAI_DEFAULTS["model"])
    resolved_temperature = (
        temperature
        if temperature is not None
        else _env_float("TEMPERATURE", OPENAI_DEFAULTS["temperature"])
    )
    _log_provider_use(
        event="chat_model_created",
        provider="openai",
        model=resolved_model,
        temperature=resolved_temperature,
    )

    return ChatOpenAI(
        model=resolved_model,
        temperature=resolved_temperature,
        **overrides,
    )


def _build_minimax(model, temperature, **overrides):
    """Build a MiniMax chat model. MiniMax exposes an OpenAI-compatible
    endpoint, so we reuse ``ChatOpenAI`` and just point it at a different
    ``base_url``. No new dependency."""
    from langchain_openai import ChatOpenAI

    class ChatMiniMaxOpenAI(ChatOpenAI):
        """ChatOpenAI tuned for MiniMax's OpenAI-compatible endpoint."""

        def with_structured_output(self, schema=None, **kwargs):
            from langchain_core.output_parsers import PydanticOutputParser
            from langchain_core.runnables import RunnableLambda

            kwargs.setdefault("method", "function_calling")
            structured = super().with_structured_output(schema, **kwargs)

            if schema is None or not hasattr(schema, "model_validate"):
                return structured

            parser = PydanticOutputParser(pydantic_object=schema)

            def invoke_with_json_fallback(input_value):
                parsed = structured.invoke(input_value)
                if parsed is not None:
                    return parsed

                _log_provider_use(
                    event="structured_output_fallback",
                    provider="minimax",
                    method="json_prompt",
                )
                prompt = (
                    "Classify the following input and return only valid JSON. "
                    "Do not include markdown, comments, or explanatory text.\n\n"
                    f"{parser.get_format_instructions()}\n\n"
                    f"Input:\n{input_value}"
                )
                raw = self.invoke(prompt).content
                try:
                    return parser.parse(raw)
                except Exception:
                    if "</think>" in raw:
                        return parser.parse(raw.split("</think>", 1)[1].strip())
                    raise

            return RunnableLambda(invoke_with_json_fallback)

    api_key = (os.getenv("MINIMAX_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "MINIMAX_API_KEY not set. "
            "Add it to your .env or export it before running. "
            "MiniMax requests cannot authenticate with OPENAI_API_KEY."
        )

    resolved_model = model or os.getenv("MINIMAX_MODEL", MINIMAX_DEFAULTS["model"])
    resolved_temperature = (
        temperature
        if temperature is not None
        else _env_float("TEMPERATURE", MINIMAX_DEFAULTS["temperature"])
    )
    resolved_base_url = (
        os.getenv("MINIMAX_BASE_URL") or MINIMAX_DEFAULTS["base_url"]
    ).strip()
    _log_provider_use(
        event="chat_model_created",
        provider="minimax",
        model=resolved_model,
        base_url=resolved_base_url,
        temperature=resolved_temperature,
    )

    return ChatMiniMaxOpenAI(
        model=resolved_model,
        temperature=resolved_temperature,
        base_url=resolved_base_url,
        api_key=api_key,
        **overrides,
    )


def get_provider_info() -> dict[str, str]:
    """Return a dict describing the active provider — useful for ``/health``
    endpoints, structured logs, and the demo that prints "which model am I
    talking to right now?".

    Returns:
        A dict with at least ``provider`` and ``model`` keys. The ``base_url``
        key is included for ``minimax`` so you can verify the endpoint.
    """
    provider = _resolve_provider()
    if provider == "minimax":
        return {
            "provider": "minimax",
            "model": os.getenv("MINIMAX_MODEL", MINIMAX_DEFAULTS["model"]),
            "base_url": os.getenv("MINIMAX_BASE_URL", MINIMAX_DEFAULTS["base_url"]),
        }
    return {
        "provider": "openai",
        "model": os.getenv("OPENAI_MODEL", OPENAI_DEFAULTS["model"]),
    }


# ── Optional: warmup-friendly lazy import shim ────────────────────────────────
# Some code paths import ``langchain.chat_models.init_chat_model`` to benefit
# from LangChain's unified registry. We expose a thin wrapper so callers can
# stay provider-aware without losing that ergonomic API.
def init_chat_model(
    model: str | None = None,
    *,
    temperature: float | None = None,
    **overrides: Any,
):
    """Drop-in replacement for ``langchain.chat_models.init_chat_model`` that
    honours ``LLM_PROVIDER``.

    The signature mirrors LangChain's: pass ``model`` (positional) and any
    kwargs. We ignore LangChain's ``model_provider`` keyword — provider is
    controlled by ``LLM_PROVIDER``.
    """
    return get_chat_model(model=model, temperature=temperature, **overrides)