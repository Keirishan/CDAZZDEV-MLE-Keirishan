"""Runtime configuration, read from environment variables.

Every value has a safe default except the OpenRouter API key. Values are looked
up in this order:
  1. Process environment variables
  2. A `.env` file in task3_agentic/ or the repository root
  3. Google Colab Secrets (when running in Colab)

Nothing in this module ever prints or logs secret values.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Environment-based settings module
# for an OpenRouter GPT-4o LangGraph project with Colab secrets fallback', Date: 2026-10-07
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# task3_agentic/  (this file is task3_agentic/src/agentic/config.py)
PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_DIR.parent

load_dotenv(PROJECT_DIR / ".env", override=False)
load_dotenv(REPO_ROOT / ".env", override=False)


class MissingAPIKeyError(RuntimeError):
    """Raised when an LLM call is attempted without OPENROUTER_API_KEY configured."""


def _colab_secret(name: str) -> str | None:
    """Read a Colab Secret if we are inside Colab and access was granted."""
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get(name)
    except Exception:  # not in Colab, secret missing, or access not granted
        return None


def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        value = _colab_secret(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = env(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = env(name)
    try:
        return float(raw) if raw is not None else default
    except ValueError:
        return default


def _resolve_path(raw: str | None, default: Path) -> Path:
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_DIR / p)


@dataclass(frozen=True)
class Settings:
    # --- LLM provider (OpenRouter, OpenAI-compatible API) ---
    openrouter_api_key: str | None
    openrouter_base_url: str
    model: str                     # main agent / report model
    sentiment_model: str           # model used inside the llm_sentiment tool
    temperature: float
    request_timeout: float
    llm_max_attempts: int
    max_output_tokens: int         # cap per LLM call (OpenRouter reserves credit for this)
    app_url: str                   # optional OpenRouter attribution headers
    app_name: str

    # --- Agent behaviour ---
    default_ticker: str
    max_tool_calls: int            # per agent, per run
    max_clarification_rounds: int  # critique loop cap (minimum one round always runs)
    recursion_limit: int

    # --- Storage ---
    trace_path: Path
    cache_dir: Path

    # --- Testing / demos ---
    fail_tools: frozenset[str]     # tool names forced to fail (failure-injection demo)

    def require_api_key(self) -> str:
        if not self.openrouter_api_key:
            raise MissingAPIKeyError(
                "OPENROUTER_API_KEY is not set. Add it to task3_agentic/.env "
                "(see .env.example) or to Colab Secrets, then restart the kernel."
            )
        return self.openrouter_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    model = env("OPENROUTER_MODEL", "openai/gpt-4o")
    return Settings(
        openrouter_api_key=env("OPENROUTER_API_KEY"),
        openrouter_base_url=env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        model=model,
        sentiment_model=env("OPENROUTER_SENTIMENT_MODEL", model),
        temperature=_env_float("LLM_TEMPERATURE", 0.1),
        request_timeout=_env_float("LLM_TIMEOUT_SECONDS", 90.0),
        llm_max_attempts=_env_int("LLM_MAX_ATTEMPTS", 4),
        max_output_tokens=_env_int("LLM_MAX_TOKENS", 2048),
        app_url=env("OPENROUTER_APP_URL", "https://github.com/"),
        app_name=env("OPENROUTER_APP_NAME", "CDAZZDEV Task 3 Agentic Research"),
        default_ticker=(env("DEFAULT_TICKER", "NVDA") or "NVDA").upper(),
        max_tool_calls=_env_int("MAX_TOOL_CALLS", 10),
        max_clarification_rounds=max(1, _env_int("MAX_CLARIFICATION_ROUNDS", 2)),
        recursion_limit=_env_int("GRAPH_RECURSION_LIMIT", 60),
        trace_path=_resolve_path(env("TRACE_PATH"), PROJECT_DIR / "logs" / "agent_trace.jsonl"),
        cache_dir=_resolve_path(env("CACHE_DIR"), PROJECT_DIR / "cache"),
        fail_tools=frozenset(
            t.strip() for t in (env("FAIL_TOOLS", "") or "").split(",") if t.strip()
        ),
    )


def reset_settings() -> None:
    """Re-read environment variables (useful in notebooks after editing os.environ)."""
    get_settings.cache_clear()
