# Agentic AI Business Analyst — Sprint Plan (Same-Day Submission)

**Assignment:** MBCIE Centre for AI — Agentic AI Data Science Assignment
**Dataset:** Olist Brazilian E-Commerce (Kaggle)
**Status:** Received 2026-08-10, due 2026-08-12 (today) — treat as a ~8-hour sprint, not a 2-day build.

## Priority tiers

Work top to bottom. Never start a P1 item with P0 items incomplete. If you run low on time, stop and move straight to Docs/Demo/Submit — a working, documented, on-time submission beats an unfinished ambitious one.

**P0 — must ship (this is literally the grading rubric)**
1. Agentic workflow with multiple tools/functions
2. SQL / data-analysis tool
3. One ML capability (forecasting OR anomaly detection — pick one, do it well)
4. A dashboard/UI to interact with the agent
5. Conversation memory across turns
6. Grounded answers (every number traceable to a real tool call)
7. Error handling (bad SQL, empty results, tool failures don't crash the app)
8. Architecture diagram + README
9. GitHub repo + 5–10 min demo recording

**P1 — do only if P0 is done with time to spare**
- Second ML capability (e.g., add anomaly detection if you did forecasting, or a customer/seller clustering)
- Nicer dashboard polish (charts, agent-trace panel)

**P2 — cut first, bonus only**
- Full multi-agent split (separate Manager / Data Analyst / ML Analyst / Business Advisor agents). See "cheap bonus" note below for a low-cost way to gesture at this without building 4 real agent loops.

## Tech stack (chosen for speed + reliability under time pressure)

| Piece | Choice | Why |
|---|---|---|
| LLM | Claude via Anthropic API (`claude-sonnet-5`), tool use / function calling | You have a key; native tool-calling means less framework glue |
| Orchestration | Hand-rolled Python loop (no LangGraph/CrewAI) | Frameworks cost setup/debug time you don't have today |
| Data store | DuckDB | Loads CSVs directly, full SQL, zero server setup |
| ML | scikit-learn + statsmodels (NOT Prophet) | Prophet has a real risk of Windows install failures eating an hour; statsmodels/sklearn are pure-Python-friendly and installed in seconds |
| Dashboard | Streamlit | Fastest path from script to working chat UI + charts |
| Charts | Plotly (via Streamlit) | Interactive, minimal code |
| Memory | `st.session_state` list of turns, replayed into each Claude call | Simplest correct implementation of "conversation state" |

## Architecture (single agent, tool-calling loop)

```
User question (Streamlit chat input)
        │
        ▼
Manager loop (Claude + tool use)
   ├─ tool: run_sql(query)            → DuckDB, schema-validated
   ├─ tool: forecast_sales(months)    → statsmodels, returns series + numbers
   ├─ tool: detect_anomalies(metric)  → z-score/IQR on monthly aggregates
   └─ tool: get_schema()              → table/column descriptions for grounding
        │
        ▼
Claude drafts answer citing only numbers that came back from a tool call
        │
        ▼
Streamlit renders: answer + evidence table/chart + which tools were called
```

**Cheap way to gesture at the bonus multi-agent design without the cost:** give the single Claude loop a system prompt that names three *internal roles* it must reason through explicitly in its response — "Data Analyst view," "ML Analyst view," "Business Advisor view" — before the final answer. It's not literally 4 separate agents, but it demonstrates the reasoning-workflow decomposition the rubric is scoring, at near-zero extra build cost. Only build *real* separate agents (P2) if P0 finishes early.

## Data plan

1. Download the 9 Olist CSVs from Kaggle (orders, order_items, customers, products, sellers, payments, reviews, category_translation, geolocation).
2. Load each into a DuckDB file (`olist.duckdb`) as-is.
3. Create 2–3 SQL views up front (`monthly_sales`, `seller_performance`, `category_performance`) so the agent has fast, correct starting points instead of writing complex joins from scratch every time — this also reduces hallucination risk.

## Grounding strategy (satisfies requirement 6 — the one graders check hardest)

- System prompt: *"Never state a number, date, or fact you did not just receive from a tool result. If you don't have the data, call a tool first."*
- Every tool result gets a short id; the final answer must reference which tool call each number came from (shown in the UI as an "evidence" expander).
- If the model tries to answer without any tool calls on a data question, the loop rejects the turn and forces a retry.

## Error handling (requirement 7)

- Wrap SQL execution in try/except; on a DuckDB error, feed the error message back to Claude as a tool result so it can self-correct the query (cap at 2 retries, then surface a clear "couldn't answer" message instead of crashing).
- Validate ML tool inputs (e.g., enough months of data to forecast) before running; return a structured error instead of throwing.

## Hour-by-hour schedule (today)

| Block | Time budget | Task |
|---|---|---|
| 1 | 45 min | Repo skeleton on GitHub, venv, install deps, download Olist CSVs, load into DuckDB, sanity-check with 2–3 manual SQL queries |
| 2 | 60 min | Build `run_sql` tool + Claude tool-use loop; get one hardcoded question answered end-to-end from the terminal |
| 3 | 45 min | Add conversation memory (session turns replayed into the loop) |
| 4 | 60 min | Build the ML tool (forecast OR anomaly detection on monthly sales) and wire it in as a second tool |
| 5 | 20 min | Add error handling / retry-on-SQL-error path |
| 6 | 60 min | Build Streamlit dashboard: chat box, answer + evidence panel, a couple of charts |
| 7 | 30 min | Run the assignment's own example question end-to-end ("Why did sales decline, and which sellers contributed most?") — verify Understand→Query→Analyze→Compare→Recommend actually shows up in the trace |
| 8 | 30 min | Test 4–5 more varied questions to confirm it's not hardcoded to one case |
| 9 | 30 min | Draw architecture diagram (mermaid → PNG), write README (setup, architecture, example Q&A, limitations) |
| 10 | 20 min | Record 5–10 min demo (screen + narration): show 2 questions, briefly point at code structure and the diagram |
| 11 | 15 min | Final commit + push, submit repo link + demo to Kumaresh.r@mbcie.org |

**Total: ~7 hours.** If you're starting later in the day than that allows, cut in this order: second ML tool → dashboard polish → number of test questions shown in the demo. Never cut grounding, error handling, or the README/diagram — those are explicit line items in the rubric.

## Deliverables checklist (map to submission)

- [ ] Working app (Streamlit, runs locally with `streamlit run app.py`)
- [ ] GitHub repo with README (setup steps, architecture explanation, example Q&A, known limitations)
- [ ] Architecture diagram (PNG/SVG, embedded in README)
- [ ] 5–10 min demo recording (uploaded, link in submission email)
- [ ] Email to Kumaresh.r@mbcie.org with repo link + demo link

## Immediate next action

Start Block 1 right now: create the GitHub repo, set up the project folder, and get the Olist dataset into DuckDB. Say go and I'll start scaffolding the repo and data pipeline.
