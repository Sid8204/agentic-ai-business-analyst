"""Streamlit dashboard for the Agentic AI Business Analyst.

Chat with a Manager agent that delegates to a Data Analyst, an ML Analyst,
and a Business Advisor to answer open-ended business questions about the
Olist e-commerce dataset, with every claim traceable back to a real tool
call.
"""

import os
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Agentic AI Business Analyst — Olist", layout="wide")

DB_PATH = Path(__file__).resolve().parent / "olist.duckdb"


def _md_safe(text: str) -> str:
    """Streamlit's markdown renderer treats lone $ as LaTeX math
    delimiters, which mangles dollar amounts in agent-generated text
    (e.g. "$6,787 and $9,495" can render as garbled math). Backslash-
    escaping ("\\$") gets silently swallowed by the renderer instead of
    showing a literal $, so a full-width dollar sign (visually identical,
    not a markdown/LaTeX special character) is substituted instead. Agent
    answers are plain business prose, never intentional LaTeX."""
    return text.replace("$", "＄")


def _check_prerequisites() -> str | None:
    if not DB_PATH.exists():
        return "Database not found. Run `python db/build_db.py` first (after downloading the Olist CSVs into data/raw/)."
    if not os.environ.get("GROQ_API_KEY"):
        return "GROQ_API_KEY is not set. Copy .env.example to .env and add a free key from console.groq.com."
    return None


error = _check_prerequisites()
if error:
    st.error(error)
    st.stop()

from agents import manager_agent, memory  # noqa: E402  (import after prerequisite check)
from agents.db import get_connection  # noqa: E402


@st.cache_data(show_spinner=False)
def dataset_overview() -> dict:
    con = get_connection()
    stats = {}
    for table in ("orders", "order_items", "customers", "sellers", "products"):
        stats[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    date_range = con.execute(
        "SELECT MIN(order_purchase_timestamp), MAX(order_purchase_timestamp) FROM orders"
    ).fetchone()
    return {"counts": stats, "date_range": date_range}


def render_charts(sub_agent_calls: list) -> None:
    for call in sub_agent_calls:
        for entry in call["tool_trace"]:
            output = entry["output"]
            if not isinstance(output, dict) or "error" in output:
                continue

            if entry["tool"] == "forecast_sales" and "chart" in output:
                chart = output["chart"]
                hist = chart["history"]
                fc = chart["forecast"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[p["month"] for p in hist], y=[p["value"] for p in hist],
                                          mode="lines+markers", name="Actual"))
                if fc:
                    bridge_x = [hist[-1]["month"]] + [p["month"] for p in fc]
                    bridge_y = [hist[-1]["value"]] + [p["value"] for p in fc]
                    fig.add_trace(go.Scatter(x=bridge_x, y=bridge_y, mode="lines+markers",
                                              name="Forecast", line=dict(dash="dash")))
                fig.update_layout(title=f"{chart['metric']} — history & forecast", height=350,
                                   margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

            elif entry["tool"] == "detect_anomalies" and output.get("anomalies") is not None:
                anomalies = output["anomalies"]
                if anomalies:
                    fig = go.Figure(go.Bar(
                        x=[a["month"] for a in anomalies],
                        y=[a["value"] for a in anomalies],
                        marker_color=["#d62728" if a["direction"] == "dip" else "#2ca02c" for a in anomalies],
                        text=[f"z={a['z_score']}" for a in anomalies],
                    ))
                    fig.update_layout(title=f"Anomalous months — {output['metric']}", height=300,
                                       margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig, use_container_width=True)

            elif entry["tool"] == "segment_customers" and output.get("segments") is not None:
                segs = output["segments"]
                fig = go.Figure(go.Bar(
                    x=[f"Cluster {s['cluster']}" for s in segs],
                    y=[s["avg_monetary"] for s in segs],
                    text=[f"{s['customer_count']} customers" for s in segs],
                ))
                fig.update_layout(title="Customer segments — avg spend", height=300,
                                   margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)


def render_trace(sub_agent_calls: list, grounding_warnings: list) -> None:
    if not sub_agent_calls:
        return
    with st.expander(f"Agent trace ({len(sub_agent_calls)} sub-agent call(s)) — how this answer was produced", expanded=False):
        for call in sub_agent_calls:
            st.markdown(f"**{call['agent']}** — _{_md_safe(call['question'][:200])}_")
            for entry in call["tool_trace"]:
                st.caption(f"→ called `{entry['tool']}`({entry['input']})")
            st.write(_md_safe(call["answer"]))
            st.divider()
        if grounding_warnings:
            st.warning(
                "Grounding check: these numbers in the final answer weren't found verbatim in "
                f"any tool output — verify before trusting them: {', '.join(grounding_warnings)}"
            )
        else:
            st.success("Grounding check passed: every number in the final answer traces back to a tool result.")
    render_charts(sub_agent_calls)


# --- Sidebar ---
with st.sidebar:
    st.title("📊 Olist Business Analyst")
    st.caption("Agentic AI Data Science Assignment — MBCIE Centre for AI")
    overview = dataset_overview()
    st.subheader("Dataset overview")
    for table, count in overview["counts"].items():
        st.metric(table, f"{count:,}")
    start, end = overview["date_range"]
    st.caption(f"Orders span {start:%Y-%m-%d} → {end:%Y-%m-%d}")
    st.divider()
    st.subheader("Architecture")
    st.caption("Manager agent → Data Analyst / ML Analyst → Business Advisor. See ARCHITECTURE.md.")
    if st.button("Clear conversation"):
        st.session_state.turns = []
        st.session_state.messages = []
        st.rerun()

# --- Chat state ---
if "turns" not in st.session_state:
    st.session_state.turns = []  # [{"question", "answer"}] for agent memory
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "sub_agent_calls", "grounding_warnings"}]

st.subheader("Ask a business question about the Olist marketplace")
st.caption('Try: "Why did sales decline, and which sellers contributed most to it?"')

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(_md_safe(msg["content"]))
        if msg["role"] == "assistant":
            render_trace(msg.get("sub_agent_calls", []), msg.get("grounding_warnings", []))

question = st.chat_input("Ask about sales trends, sellers, categories, forecasts, anomalies, customer segments...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(_md_safe(question))

    with st.chat_message("assistant"):
        with st.spinner("Manager agent is delegating to specialists..."):
            history_text = memory.format_history(st.session_state.turns)
            try:
                result = manager_agent.run(question, history_text=history_text)
            except Exception as exc:  # noqa: BLE001 — surfaced to the user, not a crash
                result = {"answer": f"Something went wrong answering that: {exc}", "sub_agent_calls": [], "grounding_warnings": []}
        st.write(_md_safe(result["answer"]))
        render_trace(result["sub_agent_calls"], result["grounding_warnings"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sub_agent_calls": result["sub_agent_calls"],
        "grounding_warnings": result["grounding_warnings"],
    })
    st.session_state.turns.append({"question": question, "answer": result["answer"]})
