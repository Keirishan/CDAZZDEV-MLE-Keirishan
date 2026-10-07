# Reflection

## Architectural decisions

**Explicit LangGraph graphs.** I built the agents as explicit `StateGraph`s rather than using `create_react_agent`. Routing, tool budgets, the critique loop and the cache shortcut are ordinary edges that can be read and tested. A custom `ToolExecutor` replaces the prebuilt `ToolNode`, so three concerns sit in one place: the per-agent allow-list, tracing and call counting. Tool restriction is therefore enforced twice. Each agent only sees its own tools, and a call outside its list is blocked and logged.

**Errors are observations.** Every tool returns a `ToolResult{ok, data, error}` envelope and never raises. A failed news fetch then becomes something the agent reasons about, typically switching to web search, instead of an exception that ends the run. The same idea applies to the LLM. Structured outputs are forced through a schema-shaped tool call and validated with Pydantic. Failures are logged and repaired once, and as a last resort a deterministic report is built from tool data and clearly marked.

**Numbers come from code.** The `DataBrief` and `ClarificationResponse` copy figures directly from tool results; the LLM only writes interpretation. In a financial setting I care more about a hallucinated RSI in a handoff than about prose quality.

**Coordination with a reason.** The brief gives Agent A `llm_sentiment` but no news, and Agent B news but no analytics. I leaned into that split. B collects headlines and attaches them to its clarification request, and A scores them and computes the 90-day volatility B needs to size the hedge. The critique loop answers a real dependency rather than existing for show. The first review must ask a question, but the model chooses what to ask.

**Memory.** Short-term memory is LangGraph's checkpointer keyed by `thread_id`. Persistent memory is a validated JSON file per ticker and day, checked by the graph's entry node. The proof is measurable: the trace's tool-line count stays unchanged across a follow-up and a cached run.

## What I would improve with more time

* **Better hedge inputs.** Options-implied volatility and real option prices would let the hedge be costed, not just placed by historical volatility.
* **Evaluation harness.** A fixed set of tickers, with an LLM-as-judge rubric for report grounding (does every number in the report appear in the trace?), would catch regressions when prompts or models change.
* **Parallel tool calls.** Independent calls such as price and volatility could run concurrently, and an async graph would cut latency.
* **Smarter caching.** Cache invalidation could follow market hours or material news, instead of the calendar date alone.
* **Better retrieval.** Embedding-based de-duplication and source-quality ranking for search results.

## Limitations encountered

* **Unreliable free data.** yfinance's news format has changed over time, so the parser handles both shapes and falls back to RSS. DuckDuckGo rate-limits bursts, which needed backoff plus a news-vertical fallback.
* **Non-determinism.** The tool order is the model's choice, so it varies between runs. Low temperature reduces this but doesn't remove it. The injected-failure demo guarantees at least one visible replan.
* **Cost.** GPT-4o is a paid OpenRouter model, while the brief prefers free tiers. The model is a single environment variable, so a free tool-calling model can be swapped in without code changes, at some cost to report quality.
* **Historical volatility.** It is backward-looking and understates event risk such as earnings.

<!-- Rewrite in your own voice before submitting; you will be asked to defend these points. -->
