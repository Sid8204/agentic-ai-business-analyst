"""Generic tool-use loop shared by every specialized agent, backed by
Groq's free-tier API (OpenAI-compatible tool calling).

Each specialized agent (Data Analyst, ML Analyst) is just this loop given a
different system prompt + tool set. The Manager and Business Advisor also
use it, but as "meta-agents" whose tools are calls into the other agents
rather than raw data tools.
"""

import json
import os
import re
import time

import groq
from groq import Groq

MODEL = "openai/gpt-oss-120b"
DEFAULT_MAX_TOOL_ROUNDS = 6

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add a free key "
                "from console.groq.com."
            )
        _client = Groq(api_key=api_key)
    return _client


def _to_function_tools(tools: list[dict]) -> list[dict]:
    """Converts our provider-agnostic {name, description, input_schema}
    tool specs into the OpenAI-style function-calling format Groq expects."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def run_agent_loop(
    system_prompt: str,
    tools: list[dict],
    dispatch: dict,
    user_content: str,
    max_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    require_tool_first: bool = False,
) -> dict:
    """Runs a full tool-use loop for one agent turn.

    require_tool_first: if True and tools are available, forces a tool call
    on the very first round (tool_choice="required") so the agent cannot
    answer a data question "from memory" without ever querying anything —
    this is the hard enforcement behind our grounding requirement, not just
    a prompt request.

    Returns {"answer": str, "trace": [{"tool": str, "input": dict, "output": dict}, ...]}
    trace is the ordered list of every tool call this agent made — the
    evidence backing whatever the answer claims.
    """
    client = get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    function_tools = _to_function_tools(tools) if tools else None
    trace = []

    for round_index in range(max_rounds):
        try:
            kwargs = dict(model=MODEL, messages=messages, temperature=0.2, max_tokens=1536)
            if function_tools:
                kwargs["tools"] = function_tools
                kwargs["tool_choice"] = "required" if (require_tool_first and round_index == 0) else "auto"
            response = _call_with_rate_limit_retry(client, kwargs)
        except groq.RateLimitError as exc:
            return {
                "answer": "The free Groq API tier's rate/token limit was hit while answering this. "
                          f"Wait a few minutes and try again. ({exc.message if hasattr(exc, 'message') else exc})",
                "trace": trace,
            }
        except groq.BadRequestError as exc:
            # Forcing tool_choice="required" (require_tool_first) can backfire: if the
            # model decides no tool call is actually needed (e.g. it can already explain
            # why a question is unanswerable), Groq rejects the response outright instead
            # of allowing plain text — but it still hands back what the model intended to
            # say in `failed_generation`. That text is usually the right answer, so use it
            # rather than surfacing a raw API error for what was actually correct behavior.
            failed_generation = None
            if isinstance(exc.body, dict):
                failed_generation = exc.body.get("error", {}).get("failed_generation")
            if failed_generation:
                return {"answer": failed_generation, "trace": trace}
            return {"answer": f"[agent error: LLM call failed — {exc}]", "trace": trace}
        except Exception as exc:  # noqa: BLE001 — surfaced to caller, not fatal to the app
            return {"answer": f"[agent error: LLM call failed — {exc}]", "trace": trace}

        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ] or None,
        })

        if not tool_calls:
            if message.content and message.content.strip():
                return {"answer": message.content, "trace": trace}
            # Blank final message with no tool call — nudge once for real
            # output instead of silently returning nothing, but only if we
            # have rounds left to spend on it.
            if round_index < max_rounds - 1:
                messages.append({
                    "role": "user",
                    "content": "Your last response was empty. Please give your final written answer now, based on the tool results already gathered.",
                })
                continue
            return {"answer": "", "trace": trace}

        for tool_call in tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            fn = dispatch.get(name)
            if fn is None:
                output = {"error": f"Unknown tool '{name}'"}
            else:
                try:
                    output = fn(**args)
                except Exception as exc:  # noqa: BLE001 — fed back to the model, not fatal
                    output = {"error": f"Tool '{name}' raised: {exc}"}

            trace.append({"tool": name, "input": args, "output": output})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": _stringify(output),
            })

    return {
        "answer": "[agent stopped: exceeded maximum tool-call rounds without a final answer]",
        "trace": trace,
    }


_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")
MAX_AUTO_RETRY_WAIT_SECONDS = 15


def _call_with_rate_limit_retry(client: Groq, kwargs: dict):
    """Groq's free tier enforces both a short per-minute rate limit and a
    much larger daily token cap. A short per-minute limit is worth an
    automatic bounded retry; a multi-minute daily-cap wait is not (it would
    just hang the app), so that case is raised immediately for the caller
    to surface to the user."""
    try:
        return client.chat.completions.create(**kwargs)
    except groq.RateLimitError as exc:
        message = str(getattr(exc, "message", exc))
        match = _RETRY_AFTER_RE.search(message)
        if match:
            minutes, seconds = match.groups()
            wait = int(minutes or 0) * 60 + float(seconds)
            if wait <= MAX_AUTO_RETRY_WAIT_SECONDS:
                time.sleep(wait)
                return client.chat.completions.create(**kwargs)
        raise


def _stringify(output) -> str:
    try:
        return json.dumps(output, default=str)
    except TypeError:
        return str(output)
