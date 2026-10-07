# Citations

## AI-assisted planning and design

Tool: **Claude (claude-opus-5-5)**, used via claude.ai, in one working session before implementation. The prompts below are quoted as written. Claude's output was a review of the brief, a plan, diagrams and a presentation aid; the candidate reviewed each step and chose the approach (LangGraph, OpenRouter with GPT-4o) before any code was written.

```
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'I have attached the PDF file that I received for a assessment. Review the document first.', Date: 2026-10-07
    -> Summary of the brief: deliverables, disqualifiers, per-task traps and a two-day schedule (not committed; informed the plan)

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'I'd like to complete all three tasks, but for now, I'm going to focus only on Task 3: Agentic Workflows. Review the problem statement and the guidelines provided in the document, and then give me a clear plan for how to build the solution, with the high level plan. I'm planning to use LangGraph because it supports multi-agent workflows and gives more control over agent orchestration, state management, and routing between agents. Consider this approach when proposing the architecture and implementation plan.', Date: 2026-10-07
    -> High-level Task 3 plan mapped to the 3A/3B/3C rubric; LangGraph architecture (choice of LangGraph made by the candidate)

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Provide the architecture diagram in Mermaid format, and also include a separate flow diagram that clearly shows the step by step process.', Date: 2026-10-07
    -> Architecture and step-by-step Mermaid diagrams (adapted into task3_agentic/README.md)

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'To make this idea easier to understand and present to others, create a simple animated visual flow. Present it task by task, and then include the complete end-to-end flow at the end. The visualization should show the process step by step. Create a simple HTML-based animated visualization where the flow moves from one stage to the next, so that it is very easy to present, understand, and explain to others.', Date: 2026-10-07
    -> Animated HTML walkthrough of 3A, 3B, 3C and the end-to-end flow (presentation aid; uses illustrative sample values, not part of the graded code)

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Before starting the implementation according to the plan and workflow we discussed earlier, give the implementation plan. Make sure the solution follows the assessment requirements and standards mentioned in the document. For the AI model integration, use OpenRouter with OpenAI GPT-4o. I want you to provide the complete codebase as a full project package. I will configure it on my local machine, add the required environment variables and API keys, and test the implementation myself. Before starting the implementation, first explain your implementation plan in detail. Once I review you can start the implementation.', Date: 2026-10-07
    -> Detailed implementation plan (requirements traceability, schemas, tools, graphs, tests), reviewed by the candidate; then the full codebase listed in the next section
```

## AI-generated code

Tool: **Claude (claude-opus-5-5)**, used via claude.ai. Architecture, the implementation plan and all design decisions were reviewed and directed by the candidate.

```
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Environment-based settings module for an OpenRouter GPT-4o LangGraph project with Colab secrets fallback', Date: 2026-10-07
    -> task3_agentic/src/agentic/config.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pydantic v2 schemas for tool results, DataBrief handoff, clarification messages and the final equity research report', Date: 2026-10-07
    -> task3_agentic/src/agentic/schemas.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'OpenRouter ChatOpenAI factory with tenacity retries and a validate-and-repair structured output helper', Date: 2026-10-07
    -> task3_agentic/src/agentic/llm.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Pandas implementations of SMA, RSI with Wilder smoothing, MACD(12,26,9), Bollinger Bands(20,2), annualised volatility, downside volatility and max drawdown', Date: 2026-10-07
    -> task3_agentic/src/agentic/indicators.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'System and user prompts for a tool-using equity research agent, a two-agent analyst/writer pipeline with a critique loop, and an LLM headline-sentiment tool', Date: 2026-10-07
    -> task3_agentic/src/agentic/prompts.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'yfinance history and info loaders with validation, retry and a same-day in-memory cache', Date: 2026-10-07
    -> task3_agentic/src/agentic/data/market.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Parse yfinance news in both the old flat and new nested content formats, with Yahoo and Google News RSS fallbacks', Date: 2026-10-07
    -> task3_agentic/src/agentic/data/news_sources.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'DuckDuckGo search wrapper using ddgs with tenacity backoff on RatelimitException and a news-search fallback', Date: 2026-10-07
    -> task3_agentic/src/agentic/data/search_backend.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Safe tool runner returning a uniform result envelope with tracing and configurable failure injection', Date: 2026-10-07
    -> task3_agentic/src/agentic/tools/base.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangChain tool wrapping yfinance that returns a compact price snapshot with SMA/RSI/MACD/Bollinger, momentum flags and fundamentals', Date: 2026-10-07
    -> task3_agentic/src/agentic/tools/price.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangChain tool computing annualised log-return volatility for a window plus 30/60/90/252-day vols, downside vol, max drawdown and the expected 90-day 1-sigma move', Date: 2026-10-07
    -> task3_agentic/src/agentic/tools/volatility.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangChain tool returning recent headlines from yfinance with RSS fallbacks as a validated list', Date: 2026-10-07
    -> task3_agentic/src/agentic/tools/news.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangChain tool that scores headlines with an LLM via a Pydantic schema and aggregates a confidence-weighted score', Date: 2026-10-07
    -> task3_agentic/src/agentic/tools/sentiment.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangChain web search tool over ddgs returning a validated list of title/snippet/url hits', Date: 2026-10-07
    -> task3_agentic/src/agentic/tools/search.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangGraph tool-executor node that enforces an allow-list, traces blocked calls, records observations and counts calls', Date: 2026-10-07
    -> task3_agentic/src/agentic/execution.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Thread-safe JSONL tracer with contextvars for agent/run/session and a timing context manager', Date: 2026-10-07
    -> task3_agentic/src/agentic/observability/tracer.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Console printer that subscribes to JSONL trace records and prints agent thoughts, tool calls and handoffs', Date: 2026-10-07
    -> task3_agentic/src/agentic/observability/printer.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Helpers for LangGraph agents: ReAct turn with printed thoughts, observation digests, budget counting and fallback report', Date: 2026-10-07
    -> task3_agentic/src/agentic/agents/common.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangGraph ReAct research agent with custom tool executor, budget-aware routing, structured final report, MemorySaver follow-ups', Date: 2026-10-07
    -> task3_agentic/src/agentic/agents/research_agent.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Analyst agent nodes for a LangGraph two-agent pipeline: ReAct loop, typed DataBrief assembly and clarification answering', Date: 2026-10-07
    -> task3_agentic/src/agentic/agents/analyst.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Writer agent nodes for a LangGraph two-agent pipeline with a guaranteed first-round critique loop and validated final report', Date: 2026-10-07
    -> task3_agentic/src/agentic/agents/writer.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'LangGraph orchestrator for analyst and writer agents with cache_check entry node, critique loop and save_cache', Date: 2026-10-07
    -> task3_agentic/src/agentic/pipeline.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'JSON cache keyed by ticker and date with atomic writes, validation on load and trace logging', Date: 2026-10-07
    -> task3_agentic/src/agentic/memory/cache.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Markdown renderer for a structured equity research report with risks, evidence and hedge sections', Date: 2026-10-07
    -> task3_agentic/src/agentic/report.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'argparse CLI for running the research agent, the two-agent pipeline and a full demo', Date: 2026-10-07
    -> task3_agentic/run.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Streamlit dashboard that reads an agent trace JSONL file and shows run filters, a timeline, per-tool durations and failures', Date: 2026-10-07
    -> task3_agentic/dashboard.py
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Scripted fake BaseChatModel supporting bind_tools and forced tool_choice for offline LangGraph tests', Date: 2026-10-07
    -> task3_agentic/tests/ (fakes.py, conftest.py, test_*.py)
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Colab notebook demonstrating every Task 3 rubric item with visible outputs', Date: 2026-10-07
    -> task3_agentic/task3_agentic.ipynb
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Draft README, CITATIONS and REFLECTION documents for the Task 3 submission', Date: 2026-10-07
    -> README.md, task3_agentic/README.md, REFLECTION.md (reflection reviewed and rewritten by the candidate)
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'When I run in the colab in the step 5 got the error check this and tell me what the issue and if any code change then only give me the code change for this only', Date: 2026-10-07
    -> task3_agentic/task3_agentic.ipynb (section 5 cell: build the per-agent tool table from the trace file, handle an empty run)
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Diagnose OpenRouter 402 errors (request requires more credits, or fewer max_tokens) and fix', Date: 2026-10-07
    -> task3_agentic/src/agentic/config.py, task3_agentic/src/agentic/llm.py, .env.example (LLM_MAX_TOKENS cap)
```

## Runtime LLM use

The system calls **`openai/gpt-4o` via OpenRouter** at runtime as the reasoning model for every agent, structured report and the `llm_sentiment` tool. The model is configured in `.env` (`OPENROUTER_MODEL`, `OPENROUTER_SENTIMENT_MODEL`). All prompts are in `task3_agentic/src/agentic/prompts.py`.

## Libraries and references (no code copied)

| Source | Used for |
|---|---|
| LangGraph docs: https://langchain-ai.github.io/langgraph/ | `StateGraph`, conditional edges, `MemorySaver` checkpointer patterns |
| LangChain docs: https://python.langchain.com/ | `@tool`, `bind_tools`, `ChatOpenAI` with a custom `base_url` |
| OpenRouter docs: https://openrouter.ai/docs | OpenAI-compatible endpoint and attribution headers |
| yfinance: https://github.com/ranaroussi/yfinance | price history, `.info`, `.get_news` |
| ddgs: https://github.com/deedy5/ddgs | DuckDuckGo text and news search |
| J. Welles Wilder, *New Concepts in Technical Trading Systems* (1978) | RSI smoothing definition |
| Gerald Appel (MACD), John Bollinger (Bollinger Bands) | indicator definitions |

<!-- Add any other AI tools you used (e.g. Copilot, Cursor) in the same format. -->
