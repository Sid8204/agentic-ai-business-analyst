# Demo recording script (5–10 min)

Record your screen with audio (Windows: Win+Alt+R for Xbox Game Bar, or
OBS Studio) with the Streamlit app running (`streamlit run app.py`) and a
code editor open in another window/tab.

## 1. Intro (30–45s)

"This is an Agentic AI Business Analyst for the Olist e-commerce dataset,
built for the MBCIE assignment. It uses four real, independent LLM agents
— a Manager, a Data Analyst, an ML Analyst, and a Business Advisor — each
with their own tools, coordinated to answer open-ended business questions
with grounded, cited answers."

Show the architecture diagram in README.md (or GitHub's rendered mermaid
view) for ~10 seconds while explaining the flow: Manager delegates to
Data Analyst / ML Analyst, then to the Business Advisor for synthesis.

## 2. The flagship question (2–3 min)

In the app, type: **"Why did sales decline, and which sellers contributed
most to it?"**

While it's thinking, narrate: "The Manager is now delegating — the Data
Analyst will find exactly which month declined and which sellers drove
it, the ML Analyst will confirm it's statistically significant, and the
Business Advisor will turn that into recommendations."

Once the answer appears:
- Read out the headline finding (which month, the % decline, top sellers).
- Expand the **"Agent trace"** panel — show the actual tool calls each
  sub-agent made (`find_largest_decline`, `compare_periods`,
  `detect_anomalies`, `forecast_sales`) and point out the **"Grounding
  check passed"** message.
- Point at the rendered anomaly chart.

## 3. A second, different-shaped question (1–2 min)

Type something ML-flavored: **"Segment our customers and tell me which
segment we should focus retention efforts on, with a sales forecast for
context."**

Narrate while it runs: "This routes to the ML Analyst for KMeans
segmentation and a forecast, then the Business Advisor turns the segment
stats into a retention recommendation."

## 4. Memory / follow-up (30s)

Ask a short follow-up that references the prior answer without restating
it, e.g. **"What about the categories in the same period?"** — point out
it correctly reuses context from the previous turn instead of asking you
to repeat yourself.

## 5. Error handling (30s, optional but strong)

Either: show a question that stresses the system (e.g. ask about "2016
vs 2017 growth") and point out the agent explicitly flags the 2016 data
as non-comparable seed data rather than reporting a misleading number —
or briefly mention it happened during development and is now handled.

## 6. Code walkthrough (1–2 min)

Switch to the editor. Show, in order:
1. `agents/manager_agent.py` — the delegation tools + system prompt.
2. `agents/tools_sql.py` — `run_sql`'s read-only validation and the
   `find_largest_decline`/`compare_periods` analysis tools.
3. `agents/tools_ml.py` — one ML tool (e.g. `detect_anomalies`).
4. `agents/base_agent.py` — the grounding/`require_tool_first` enforcement
   and rate-limit handling.

## 7. Close (15s)

"Everything's grounded in real tool calls, errors are handled rather than
crashing, and the full source, README, and architecture diagram are in
the GitHub repo." Show the repo URL on screen.

---

**Repo:** https://github.com/Sid8204/agentic-ai-business-analyst
**Submit to:** Kumaresh.r@mbcie.org — repo link + demo video link.
