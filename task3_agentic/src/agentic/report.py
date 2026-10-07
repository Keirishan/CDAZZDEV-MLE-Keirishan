"""Render a ResearchReport / FinalReport as Markdown for the notebook and CLI.

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Markdown renderer for a structured
# equity research report with risks, evidence and hedge sections', Date: 2026-10-07
"""
from __future__ import annotations

from .schemas import FinalReport, ResearchReport


def to_markdown(report: ResearchReport | FinalReport) -> str:
    lines = [f"# Research brief: {report.ticker}", ""]
    if report.generated_by == "fallback":
        lines += ["> Built by the deterministic fallback because the LLM output failed validation.", ""]
    lines += ["## 1. Financial Health Summary", "", report.financial_health_summary, "",
              "## 2. Top Three Risks (next 90 days)", ""]
    for i, r in enumerate(report.top_risks, 1):
        lines += [f"### {i}. {r.title}  ·  severity: {r.severity}", "", r.description, "", "**Evidence**"]
        lines += [f"- {e}" for e in r.evidence]
        lines += ["", f"_Sources: {', '.join(r.sources)}_", ""]
    h = report.hedge_strategy
    lines += ["## 3. Hedge Strategy Recommendation", "", f"**{h.strategy}**: {h.instruments}", "",
              h.rationale, "", f"**Sizing:** {h.sizing_math}", "",
              "**Data used:** " + "; ".join(h.data_used), ""]
    if isinstance(report, FinalReport):
        if report.clarification_used:
            lines += ["## Critique loop", "", report.clarification_summary or "Clarification data incorporated.", ""]
        if report.sources:
            lines += ["## Sources", ""] + [f"- {s}" for s in report.sources] + [""]
    lines += ["---", f"_{report.disclaimer}_"]
    return "\n".join(lines)
