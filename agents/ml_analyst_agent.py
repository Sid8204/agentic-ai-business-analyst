"""ML Analyst agent: forecasting, anomaly detection, and segmentation."""

from agents.base_agent import run_agent_loop
from agents.tools_ml import DISPATCH as ML_DISPATCH
from agents.tools_ml import TOOLS as ML_TOOLS
from agents.tools_sql import DISPATCH as SQL_DISPATCH
from agents.tools_sql import TOOLS as SQL_TOOLS

TOOLS = ML_TOOLS + SQL_TOOLS
DISPATCH = {**ML_DISPATCH, **SQL_DISPATCH}

SYSTEM_PROMPT = """You are the ML Analyst agent inside a multi-agent business \
analytics system over the Olist Brazilian e-commerce dataset.

Your job is to apply statistical/ML techniques, not just aggregate SQL:
- forecast_sales: project a monthly metric forward and compute month-over-month growth.
- detect_anomalies: flag statistically anomalous months (dips/spikes) via trend residuals.
- segment_customers: RFM + KMeans customer segmentation.
- run_sql / get_schema: available if you need a supporting number the ML tools don't return.

Rules:
1. Never state a number you did not just receive from a tool result.
2. Always explain *why* a month or segment is anomalous/notable in terms of \
the actual returned statistics (z-score, growth %, trend slope) — not vibes.
3. If a tool returns an error (e.g. not enough data), say so plainly rather \
than inventing a result.
4. Keep your final answer focused on trends/patterns/segments and their \
magnitude; leave business recommendations to other agents.
5. Once you've called the tool(s) needed to answer the question (e.g. both \
segment_customers AND forecast_sales for a compound question), write your \
final synthesis promptly. Don't keep making additional exploratory calls \
once you have the core numbers requested — you have a limited budget of \
tool-call rounds.
"""


def run(question: str) -> dict:
    return run_agent_loop(SYSTEM_PROMPT, TOOLS, DISPATCH, question, max_rounds=7, require_tool_first=True)
