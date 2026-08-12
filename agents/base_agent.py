"""Generic Claude tool-use loop shared by every specialized agent.

Each specialized agent (Data Analyst, ML Analyst) is just this loop given a
different system prompt + tool set. The Manager and Business Advisor also
use it, but as "meta-agents" whose tools are calls into the other agents
rather than raw data tools.
"""

import os

import anthropic

MODEL = "claude-sonnet-5"
MAX_TOOL_ROUNDS = 6

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def run_agent_loop(system_prompt: str, tools: list[dict], dispatch: dict, user_content: str) -> dict:
    """Runs a full tool-use loop for one agent turn.

    Returns {"answer": str, "trace": [{"tool": str, "input": dict, "output": dict}, ...]}
    trace is the ordered list of every tool call this agent made — the
    evidence backing whatever the answer claims.
    """
    client = get_client()
    messages = [{"role": "user", "content": user_content}]
    trace = []

    for _ in range(MAX_TOOL_ROUNDS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )
        except anthropic.APIError as exc:
            return {"answer": f"[agent error: LLM call failed — {exc}]", "trace": trace}

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            return {"answer": text, "trace": trace}

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            fn = dispatch.get(block.name)
            if fn is None:
                output = {"error": f"Unknown tool '{block.name}'"}
            else:
                try:
                    output = fn(**block.input)
                except Exception as exc:  # noqa: BLE001 — fed back to the model, not fatal
                    output = {"error": f"Tool '{block.name}' raised: {exc}"}
            trace.append({"tool": block.name, "input": block.input, "output": output})
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": _stringify(output)}
            )
        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": "[agent stopped: exceeded maximum tool-call rounds without a final answer]",
        "trace": trace,
    }


def _stringify(output) -> str:
    import json

    try:
        return json.dumps(output, default=str)
    except TypeError:
        return str(output)
