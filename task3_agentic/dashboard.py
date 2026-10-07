"""Streamlit trace dashboard for logs/agent_trace.jsonl (Task 3C bonus).

Run:  streamlit run dashboard.py

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Streamlit dashboard that reads an agent
# trace JSONL file and shows run filters, a timeline, per-tool durations and failures',
# Date: 2026-10-07
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

DEFAULT_TRACE = Path(__file__).resolve().parent / "logs" / "agent_trace.jsonl"

st.set_page_config(page_title="Agent Trace", layout="wide")
st.title("Agent trace viewer")

path = Path(st.sidebar.text_input("Trace file", str(DEFAULT_TRACE)))
if not path.exists():
    st.warning(f"No trace file at {path}. Run the notebook or `python run.py demo` first.")
    st.stop()

rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
df = pd.DataFrame(rows)
df["ts"] = pd.to_datetime(df["ts"])

runs = sorted(df["run_id"].unique(), key=lambda r: df.loc[df.run_id == r, "ts"].min(), reverse=True)
run = st.sidebar.selectbox("Run", ["(all)"] + runs)
kinds = st.sidebar.multiselect("Record kinds", sorted(df["kind"].unique()), default=["tool", "guard", "cache"])
view = df if run == "(all)" else df[df.run_id == run]
view = view[view.kind.isin(kinds)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records", len(view))
c2.metric("Tool calls", int((view.kind == "tool").sum()))
c3.metric("Failures / blocks", int((~view.status.isin(["ok", "hit", "miss", "saved"])).sum()))
c4.metric("Total tool time (s)", f"{view.loc[view.kind == 'tool', 'duration_ms'].sum() / 1000:.1f}")

st.subheader("Timeline")
st.dataframe(view[["ts", "agent", "kind", "tool", "status", "duration_ms", "output"]].sort_values("ts"),
             use_container_width=True, hide_index=True)

tools = view[view.kind == "tool"]
if not tools.empty:
    left, right = st.columns(2)
    with left:
        st.subheader("Mean duration by tool (ms)")
        st.bar_chart(tools.groupby("tool")["duration_ms"].mean())
    with right:
        st.subheader("Calls by agent and tool")
        st.dataframe(tools.groupby(["agent", "tool"]).size().rename("calls").reset_index(),
                     use_container_width=True, hide_index=True)

bad = view[~view.status.isin(["ok", "hit", "miss", "saved"])]
if not bad.empty:
    st.subheader("Failures, blocked calls and validation errors")
    st.dataframe(bad[["ts", "agent", "tool", "status", "output", "args"]], use_container_width=True, hide_index=True)
