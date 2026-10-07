"""Command-line entry point for local testing.

Examples:
    python run.py agent --ticker NVDA                    # Task 3A
    python run.py agent --ticker NVDA --fail get_news    # 3A with injected tool failure
    python run.py agent --ticker NVDA --followup "What RSI did you retrieve earlier?"
    python run.py pipeline --ticker NVDA --fresh         # Task 3B (ignore cache)
    python run.py pipeline --ticker NVDA                 # second run -> cache hit (3C)
    python run.py demo --ticker NVDA                     # everything, in notebook order

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'argparse CLI for running the research
# agent, the two-agent pipeline and a full demo', Date: 2026-10-07
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agentic import ResearchAgent, ResearchPipeline, get_settings, to_markdown, tracer  # noqa: E402
from agentic.config import MissingAPIKeyError  # noqa: E402
from agentic.tools import failing  # noqa: E402


def cmd_agent(args: argparse.Namespace) -> None:
    agent = ResearchAgent()
    if args.fail:
        with failing(*args.fail):
            res = agent.run(args.ticker)
    else:
        res = agent.run(args.ticker)
    print("\n" + to_markdown(res.report))
    print(f"\nTool sequence chosen by the agent: {res.tool_sequence}")
    if args.followup:
        fu = agent.ask(args.followup, session=res.session)
        print(f"\nFollow-up answer:\n{fu.answer}\nAnswered from memory: {fu.answered_from_memory}")


def cmd_pipeline(args: argparse.Namespace) -> None:
    out = ResearchPipeline().run(args.ticker, force_refresh=args.fresh)
    print("\n" + to_markdown(out.final_report))
    print(f"\ncache_hit={out.cache_hit}  tool_calls_this_run={out.tool_calls}  "
          f"clarification_rounds={len(out.clarifications)}  cache_file={out.cache_path}")


def cmd_demo(args: argparse.Namespace) -> None:
    agent = ResearchAgent()
    res = agent.run(args.ticker)
    print("\n" + to_markdown(res.report))
    with failing("get_news"):
        agent.run(args.ticker)
    fu = agent.ask("What RSI value did you retrieve earlier, and which tool did it come from?", session=res.session)
    print(f"\nFollow-up: {fu.answer}\nAnswered from memory: {fu.answered_from_memory}")
    pipe = ResearchPipeline()
    first = pipe.run(args.ticker, force_refresh=True)
    print("\n" + to_markdown(first.final_report))
    second = pipe.run(args.ticker)
    print(f"\nSecond run cache_hit={second.cache_hit}, tool calls={second.tool_calls}")


def main() -> int:
    p = argparse.ArgumentParser(description="CDAZZDEV Task 3 agentic research system")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("agent", help="Task 3A single research agent")
    a.add_argument("--ticker", default=None)
    a.add_argument("--fail", nargs="*", default=[], help="tool names to force-fail, e.g. get_news")
    a.add_argument("--followup", default=None, help="follow-up question answered from memory")
    a.set_defaults(fn=cmd_agent)
    b = sub.add_parser("pipeline", help="Task 3B two-agent pipeline with cache")
    b.add_argument("--ticker", default=None)
    b.add_argument("--fresh", action="store_true", help="ignore today's cached brief")
    b.set_defaults(fn=cmd_pipeline)
    d = sub.add_parser("demo", help="run every demo in notebook order")
    d.add_argument("--ticker", default=None)
    d.set_defaults(fn=cmd_demo)
    args = p.parse_args()
    args.ticker = (args.ticker or get_settings().default_ticker).upper()
    try:
        args.fn(args)
    except MissingAPIKeyError as exc:
        print(f"\nConfiguration error: {exc}", file=sys.stderr)
        return 2
    print(f"\nTrace file: {tracer.path} ({tracer.count('tool')} tool records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
