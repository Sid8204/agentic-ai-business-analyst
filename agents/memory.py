"""Conversation memory helper. The dashboard keeps the full turn history in
Streamlit session_state; this just formats the last few turns into a text
block the Manager agent can use for context.
"""

MAX_TURNS_IN_CONTEXT = 5


def format_history(turns: list[dict]) -> str:
    """turns: list of {"question": str, "answer": str}, oldest first."""
    if not turns:
        return ""
    recent = turns[-MAX_TURNS_IN_CONTEXT:]
    lines = []
    for i, turn in enumerate(recent, 1):
        lines.append(f"Q{i}: {turn['question']}\nA{i}: {turn['answer']}")
    return "\n\n".join(lines)
