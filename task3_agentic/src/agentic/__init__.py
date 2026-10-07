"""CDAZZDEV Task 3: multi-agent financial research system built with LangGraph.

Quick start:
    from agentic import ResearchAgent, run_pipeline
    res = ResearchAgent().run("NVDA")          # Task 3A
    out = run_pipeline("NVDA")                  # Task 3B + 3C
"""
from .agents import FollowupResult, ResearchAgent, ResearchRunResult
from .config import get_settings, reset_settings
from .observability import console, tracer
from .pipeline import PipelineResult, ResearchPipeline, run_pipeline
from .report import to_markdown

__all__ = [
    "ResearchAgent", "ResearchRunResult", "FollowupResult",
    "ResearchPipeline", "PipelineResult", "run_pipeline",
    "get_settings", "reset_settings", "console", "tracer", "to_markdown",
]
__version__ = "1.0.0"
