import json
from datetime import date

from agentic.execution import ToolExecutor
from agentic.memory.cache import cache_path, load_cached, save_cached
from agentic.observability.tracer import trace_call, tracer, truncate
from agentic.schemas import CachedBrief, FinalReport
from agentic.tools import WRITER_TOOLS

REQUIRED_FIELDS = {"ts", "run_id", "session", "agent", "kind", "tool", "args", "output", "duration_ms", "status"}


def test_trace_record_fields_and_truncation():
    with trace_call("demo_tool", {"x": 1}) as span:
        span.finish("y" * 500)
    rec = tracer.read()[-1]
    assert REQUIRED_FIELDS <= set(rec)
    assert len(rec["output"]) == 200 and rec["output"].endswith("...")
    assert len(truncate("short")) == 5


def test_guard_blocks_disallowed_tool():
    ex = ToolExecutor("writer", WRITER_TOOLS, messages_key="writer_messages", phase="research")
    msgs, obs, executed = ex.execute_calls([{"name": "get_price_data", "args": {"ticker": "NVDA"}, "id": "c1"}])
    assert executed == 0 and obs[0]["executed"] is False
    assert "not permitted" in json.loads(msgs[0].content)["error"]
    assert tracer.read()[-1]["status"] == "blocked"


def test_invalid_arguments_are_returned_to_agent():
    ex = ToolExecutor("writer", WRITER_TOOLS)
    msgs, _, _ = ex.execute_calls([{"name": "get_news", "args": {"ticker": "NVDA", "n": 999}, "id": "c2"}])
    assert "Invalid arguments" in json.loads(msgs[0].content)["error"]


def _report():
    risk = {"title": "r", "description": "d", "evidence": ["a 1", "b 2"], "sources": ["x"], "severity": "low"}
    return FinalReport(ticker="NVDA", financial_health_summary="ok", top_risks=[risk] * 3,
                       hedge_strategy={"strategy": "put", "instruments": "i", "rationale": "r",
                                       "sizing_math": "m", "data_used": ["v"]})


def test_cache_roundtrip_and_corrupt_file():
    brief = CachedBrief(ticker="NVDA", date=date.today().isoformat(), model="m", run_id="r",
                        tool_call_count=5, data_brief=None, final_report=_report())
    path = save_cached(brief)
    assert path == cache_path("NVDA") and load_cached("NVDA").tool_call_count == 5
    path.write_text("{not json")
    assert load_cached("NVDA") is None
    assert tracer.read()[-1]["status"] == "corrupt"


def test_report_requires_exactly_three_risks():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        FinalReport(ticker="X", financial_health_summary="s", top_risks=[],
                    hedge_strategy={"strategy": "p", "instruments": "i", "rationale": "r",
                                    "sizing_math": "m", "data_used": ["v"]})
