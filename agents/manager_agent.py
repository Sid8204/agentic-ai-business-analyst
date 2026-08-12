"""Manager agent: the orchestrator. Understands the user's question,
delegates to the Data Analyst and/or ML Analyst, then delegates synthesis
to the Business Advisor. This is the entry point the dashboard calls.
"""

import re

from agents import business_advisor_agent, data_analyst_agent, ml_analyst_agent
from agents.base_agent import run_agent_loop

SYSTEM_PROMPT = """You are the Manager agent coordinating a team of three \
specialists to answer business questions about the Olist e-commerce dataset:
- Data Analyst: answers factual/quantitative questions via SQL.
- ML Analyst: forecasting, anomaly detection, customer segmentation.
- Business Advisor: synthesizes findings into insights + recommendations.

You have NO direct data access — you must delegate. Typical flow for an \
open-ended business question (e.g. "why did sales decline, and which \
sellers contributed most?"):
1. Understand the question.
2. Call ask_data_analyst for the relevant facts/comparisons (e.g. which \
month declined, which sellers/categories drove it).
3. Call ask_ml_analyst if trend, forecasting, anomaly, or segmentation \
analysis would strengthen the answer.
4. Call ask_business_advisor, passing it the original question plus a \
concise summary of what the Data Analyst and ML Analyst found, to get the \
final synthesized answer with recommendations.
5. Present the Business Advisor's synthesis as your final answer. You may \
lightly reformat it, but must NOT add new numbers that didn't come from \
your specialists.

For simple factual questions you may skip straight to ask_data_analyst and \
answer directly without invoking the Business Advisor. Use judgment — but \
when the question asks "why" or "what should we do", always route through \
the Business Advisor before answering.

For decline/root-cause/trend-attribution questions specifically, phrase \
your ask_data_analyst request as an explicit two-step instruction so the \
Data Analyst doesn't have to guess a strategy: "First call \
find_largest_decline to identify the exact month and prior month. Then \
call compare_periods with dimension='seller' (and separately \
dimension='category') using those two months to find the top decliners." \
Do this in ONE ask_data_analyst call, not several.

IMPORTANT — do not re-ask a specialist the same or a reworded question \
more than once, even if its answer seems incomplete or says it ran out of \
budget. Work with whatever findings you do have and note any gap to the \
Business Advisor rather than re-delegating — re-asking wastes your \
remaining turns and rarely produces a different result. You have a \
limited number of turns — budget them as: one ask_data_analyst call, \
optionally one ask_ml_analyst call, then one ask_business_advisor call, \
then answer.

You also maintain conversation memory: earlier turns in this conversation \
are included below. Use them for context (e.g. "those sellers" may refer to \
a previous answer), but always re-verify current figures via your \
specialists rather than reusing stale numbers from memory.
"""

_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*%?")


def _make_tools_and_dispatch(sub_agent_calls: list):
    def ask_data_analyst(question: str):
        result = data_analyst_agent.run(question)
        sub_agent_calls.append(
            {"agent": "Data Analyst", "question": question, "answer": result["answer"], "tool_trace": result["trace"]}
        )
        return {"findings": result["answer"]}

    def ask_ml_analyst(question: str):
        result = ml_analyst_agent.run(question)
        sub_agent_calls.append(
            {"agent": "ML Analyst", "question": question, "answer": result["answer"], "tool_trace": result["trace"]}
        )
        return {"findings": result["answer"]}

    def ask_business_advisor(context: str):
        result = business_advisor_agent.run(context)
        sub_agent_calls.append(
            {"agent": "Business Advisor", "question": context, "answer": result["answer"], "tool_trace": result["trace"]}
        )
        return {"synthesis": result["answer"]}

    tools = [
        {
            "name": "ask_data_analyst",
            "description": "Delegate a factual/quantitative question to the Data Analyst agent (SQL over the Olist dataset).",
            "input_schema": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
        {
            "name": "ask_ml_analyst",
            "description": "Delegate a forecasting/anomaly-detection/segmentation question to the ML Analyst agent.",
            "input_schema": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
        {
            "name": "ask_business_advisor",
            "description": "Hand the original question plus a summary of gathered findings to the Business Advisor for synthesis and recommendations.",
            "input_schema": {
                "type": "object",
                "properties": {"context": {"type": "string", "description": "Original question + summarized findings from other agents."}},
                "required": ["context"],
            },
        },
    ]
    dispatch = {
        "ask_data_analyst": lambda **kw: ask_data_analyst(**kw),
        "ask_ml_analyst": lambda **kw: ask_ml_analyst(**kw),
        "ask_business_advisor": lambda **kw: ask_business_advisor(**kw),
    }
    return tools, dispatch


def _check_grounding(answer: str, sub_agent_calls: list) -> list[str]:
    """Heuristic anti-hallucination check: every number >= 10 (or with a
    decimal/percent) in the final answer should appear somewhere in the
    evidence the sub-agents actually returned."""
    evidence_blob = " ".join(call["answer"] for call in sub_agent_calls)
    for call in sub_agent_calls:
        for entry in call["tool_trace"]:
            evidence_blob += " " + str(entry["output"])

    warnings = []
    for match in _NUMBER_RE.findall(answer):
        cleaned = match.replace(",", "").rstrip("%")
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if abs(value) < 10:
            continue  # too likely to be a list index / small count, not worth flagging
        if match not in evidence_blob and cleaned not in evidence_blob:
            warnings.append(match)
    return warnings


def run(user_question: str, history_text: str = "") -> dict:
    sub_agent_calls: list = []
    tools, dispatch = _make_tools_and_dispatch(sub_agent_calls)

    user_content = user_question
    if history_text:
        user_content = f"Conversation so far:\n{history_text}\n\nCurrent question: {user_question}"

    result = run_agent_loop(SYSTEM_PROMPT, tools, dispatch, user_content, max_rounds=6, require_tool_first=True)
    warnings = _check_grounding(result["answer"], sub_agent_calls)

    return {
        "answer": result["answer"],
        "sub_agent_calls": sub_agent_calls,
        "manager_trace": result["trace"],
        "grounding_warnings": warnings,
    }
