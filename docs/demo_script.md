# Video script

This is a speaking script built around one continuous use case: the sales decline question from the assignment brief. It runs about six to seven minutes if read at a normal pace, comfortably inside the five to ten minute window. Read it in your own words rather than word for word if that feels more natural on camera, the structure and the beats are the important part.

Before you record, have the following open and ready:

* The Streamlit app running in a browser tab, at a fresh page load with no earlier conversation in it.
* The GitHub repository open in another tab, scrolled to the architecture diagram in the README.
* A code editor with `agents/manager_agent.py`, `agents/tools_sql.py`, and `agents/base_agent.py` easy to switch between.

## 1. Introduction (about 30 seconds)

Start on the README with the architecture diagram visible.

"Hi, I'm Siddharth. This is my submission for the agentic AI business analyst assignment. The idea is a system that can take an open ended business question about the Olist ecommerce dataset and actually go figure out the answer itself: query the data, run some analysis, and explain its reasoning, rather than returning a canned lookup."

## 2. Walking through the architecture (about 45 seconds)

Point at the diagram while you talk.

"I built this as a small team of agents rather than one model doing everything. A Manager agent reads the question and decides who needs to get involved. A Data Analyst agent handles anything that needs a database query. An ML Analyst agent runs forecasting, anomaly detection, or customer segmentation when that's relevant. And a Business Advisor agent takes what the other two found and turns it into an answer with actual recommendations.

The important part is that the Manager has no direct access to the data at all. It can only learn things by asking one of the specialists, so there's no way for it to just make a number up. And the Business Advisor has no tools either, so it can only work with evidence it was actually handed."

## 3. Running the use case (about three minutes)

Switch to the app.

"Let me ask it the exact question from the assignment brief."

Type into the chat: **Why did sales decline, and which sellers contributed most to it?**

While it is processing:

"While that's running, here's what's actually happening behind the scenes. The Manager is asking the Data Analyst to find the month with the sharpest drop and which sellers were behind it, and asking the ML Analyst to check whether that drop is statistically meaningful or just normal noise. Once both come back, it hands everything to the Business Advisor to turn into an actual answer."

When the answer appears, read the key parts out loud in your own words: which month declined, the size of the drop, the top sellers involved, and the recommendations at the end. Then say:

"Now let me show you how this answer was actually produced, not just what it says."

Click to expand the **Agent trace** panel.

"Every single one of these is a real tool call with real arguments and a real result. You can see the Data Analyst called a tool that finds the largest month over month decline, then compared seller revenue between those two exact months. The ML Analyst ran anomaly detection on the same trend and a short term forecast. And down here it says the grounding check passed, meaning every number in the final answer was traced back to one of these actual results. If a number didn't match anything a tool returned, this would flag it instead of silently trusting it."

Point at the anomaly chart if one rendered.

"And this chart isn't decorative, it's generated directly from the same tool output."

## 4. Showing memory (about 30 seconds)

Type a short follow up that leans on the previous answer without restating it, for example: **What about the categories in that same period?**

"Notice I didn't have to repeat the month or explain what I meant. It remembers the conversation and pulls the right context automatically."

Read the new answer briefly once it appears.

## 5. A quick look at the code (about a minute)

Switch to the editor.

"Just to show this isn't a black box, here's the Manager agent's system prompt. It's explicitly told it has no data access and must delegate everything." (Show `manager_agent.py`.)

"Here's the SQL tool. It only accepts read only statements, and if a query fails it feeds the error back to the model so it can correct itself instead of just giving up." (Show `tools_sql.py`.)

"And this is the shared tool use loop every agent runs on. It's also where I handle rate limits and a couple of edge cases I ran into with the model occasionally deciding it didn't need a tool when it actually had a perfectly good answer without one." (Show `base_agent.py`.)

## 6. Closing (about 20 seconds)

Back to the app or the GitHub repo.

"So that covers the core requirements: a real agentic workflow with multiple tools, a SQL and analysis layer, forecasting and anomaly detection and segmentation for the ML side, a working dashboard, memory across turns, and grounding checks so the answers stay honest. The full source code, the README, and this recording are all in the repository. Thanks for watching."

Show the repo URL clearly on screen for a couple of seconds before cutting.

---

Repository: https://github.com/Sid8204/agentic-ai-business-analyst
