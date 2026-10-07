"""End-to-end graph tests with the scripted fake LLM (offline)."""
from agentic import ResearchAgent, ResearchPipeline, to_markdown
from agentic.observability.tracer import tracer
from agentic.tools import failing


def test_research_agent_recovers_from_tool_failure_and_repairs_report(brain):
    agent = ResearchAgent()
    with failing("get_news"):
        res = agent.run("NVDA", session="t-3a")
    seq = res.tool_sequence
    assert seq[:4] == ["get_price_data", "calculate_volatility", "get_news", "web_search"]
    news_obs = [o for o in res.observations if o["tool"] == "get_news"][0]
    assert news_obs["result"]["ok"] is False                 # failure observed
    assert len(res.report.top_risks) == 3                    # repaired after invalid first output
    assert brain.structured_calls["ResearchReport"] == 2
    assert any(r["kind"] == "validation" for r in tracer.read())
    assert "Top Three Risks" in to_markdown(res.report)


def test_followup_is_answered_from_memory(brain):
    agent = ResearchAgent()
    agent.run("NVDA", session="t-mem")
    fu = agent.ask("What RSI did you retrieve earlier?", session="t-mem")
    assert fu.answered_from_memory
    assert "RSI" in fu.answer


def test_pipeline_critique_loop_restriction_and_cache(brain):
    pipe = ResearchPipeline()
    out = pipe.run("NVDA", session="t-3b", force_refresh=True)

    # Tool restriction: each agent only used its own tools
    assert {o["tool"] for o in out.observations if o["agent"] == "writer"} <= {"web_search", "get_news"}
    assert {o["tool"] for o in out.observations if o["agent"] == "analyst"} <= {
        "get_price_data", "calculate_volatility", "llm_sentiment"}

    # Critique loop ran once and was incorporated
    assert len(out.clarifications) == 1
    req = out.clarifications[0]["request"]
    assert req["headlines"], "writer's headlines must be attached for the analyst"
    assert out.final_report.clarification_used is True
    assert out.data_brief.sentiment is not None              # filled in during clarification

    # Structured handoff
    assert out.data_brief.price.sma200 is not None

    # Persistent cache: second run loads the file and calls no tools
    first_calls = out.tool_calls
    again = pipe.run("NVDA", session="t-3b-2")
    assert again.cache_hit is True and again.tool_calls == 0 and first_calls > 0
    assert again.final_report.ticker == "NVDA"
    assert tracer.count("tool", run_id=again.run_id) == 0
