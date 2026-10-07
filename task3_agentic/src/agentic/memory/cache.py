"""Persistent memory: final research briefs saved as JSON keyed by ticker and date.

File name: cache/{TICKER}_{YYYY-MM-DD}.json (local date). A file that fails
validation is treated as a cache miss, never as a crash.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'JSON cache keyed by ticker and date with
# atomic writes, validation on load and trace logging', Date: 2026-10-07
"""
from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from ..config import get_settings
from ..observability.tracer import tracer
from ..schemas import CachedBrief


def cache_path(ticker: str, on: date | None = None) -> Path:
    return get_settings().cache_dir / f"{ticker.upper()}_{(on or date.today()).isoformat()}.json"


def load_cached(ticker: str, on: date | None = None) -> CachedBrief | None:
    path = cache_path(ticker, on)
    t0 = time.perf_counter()
    status, out, brief = "miss", f"{path.name} not found", None
    if path.exists():
        try:
            brief = CachedBrief.model_validate_json(path.read_text(encoding="utf-8"))
            status, out = "hit", f"loaded {path.name}"
        except (ValidationError, json.JSONDecodeError, OSError) as exc:
            status, out = "corrupt", f"{path.name} unreadable ({type(exc).__name__}); treating as miss"
    tracer.log(kind="cache", tool="cache_lookup", args={"ticker": ticker.upper(), "path": path.name},
               output=out, duration_ms=(time.perf_counter() - t0) * 1000, status=status)
    return brief


def save_cached(brief: CachedBrief) -> Path:
    path = cache_path(brief.ticker, date.fromisoformat(brief.date))
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(brief.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic on the same filesystem
    tracer.log(kind="cache", tool="cache_save", args={"ticker": brief.ticker, "path": path.name},
               output=f"saved {path.name}", duration_ms=(time.perf_counter() - t0) * 1000, status="saved")
    return path


def delete_cached(ticker: str, on: date | None = None) -> bool:
    path = cache_path(ticker, on)
    if path.exists():
        path.unlink()
        return True
    return False
