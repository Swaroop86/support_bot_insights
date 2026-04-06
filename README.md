# Identity Support Bot Analyzer

A Streamlit web app that analyzes AI support bot performance data and generates an interactive HTML report. Built for the Identity team to understand escalation patterns, root causes, and improvement opportunities.

---

## Live App

> Deploy to Streamlit Cloud and paste your URL here once live.

---

## What It Does

Upload a CSV export from your support bot platform and the app will:

- Compute **weekly resolution rates** and trend charts
- Break down escalations by **intent category** (how-to, troubleshooting, access request, notification)
- Run **root cause analysis** on every escalated thread using orchestration trace signals
- **Cluster similar threads** using TF-IDF + KMeans to surface repeating patterns
- Analyze **engineer resolution messages** to identify automation opportunities
- Generate a **prioritized recommendations list** split by Bot Platform vs Capability Team
- Package everything into a **self-contained ~4–17 MB HTML report** you can open in any browser

---

## Quickstart (Local)

```bash
# Clone the repo
git clone <your-repo-url>
cd bot_insights_guide

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501), upload your CSV, and click **Generate Report**.

To run the analysis script directly (no UI):

```bash
cd data
python3 analyze_v4.py
# Output: data/support_bot_report_v4.html
```

---

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app** → select your repo and branch
4. Set **Main file path** to `app.py`
5. Click **Deploy**

Your app will be live at `https://<app-name>.streamlit.app` in ~2 minutes.

---

## Expected CSV Format

The CSV should be a thread-level export from the identity-support Slack channel. Required columns:

| Column | Type | Description |
|--------|------|-------------|
| `threadid` | string | Unique thread identifier |
| `escalated` | boolean | TRUE if thread was escalated to a human |
| `resolution_type` | string | `BOT` or `HUMAN` |
| `intent_category` | string | `howTo`, `troubleshooting`, `accessRequest`, `notification` |
| `conversation_history` | JSON string | Full Slack thread messages |
| `ai_orchestrator_trace` | JSON string | Bot orchestration trace with llm_status events |
| `week_start` | date | ISO date of the week start |
| `faq_rendered` | boolean | Whether an FAQ card was shown |
| `prefiltered` | boolean | Whether thread was caught by prefilter rules |
| `prefilter_action` | string | Action taken by prefilter |
| `rca_human_available` | boolean | Whether a human-completed RCA survey exists |
| `rca_llm_root_cause_tag` | string | LLM-assigned root cause tag |

---

## Project Structure

```
bot_insights_guide/
├── app.py                     # Streamlit web app
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── CLAUDE.md                  # Developer guide for Claude Code
└── data/
    ├── analyze_v4.py          # Full analysis pipeline
    └── support_bot_report_v4.html   # Generated report (git-ignored)
```

---

## Report Sections

The generated HTML report contains 14 sections:

1. **Executive Summary** — KPI cards + resolution rate trend chart
2. **Weekly Breakdown** — clickable rows with intent deep dive per week
3. **Intent Overview** — bar + donut charts by intent category
4. **Intent by Week (6-Wk)** — per-intent tab showing all weeks
5. **Weekly Deep Dive** — per-week tab with full cluster analysis
6. **How-To Deep Dive** — 5 sub-categories with thread-level detail
7. **Troubleshooting Clusters** — 8 clusters with engineer resolution analysis
8. **Access Request Clusters** — 8 clusters
9. **Notification/Enquiry Clusters** — 7 clusters
10. **Thread Explorer** — searchable/filterable table of all threads
11. **Recommendations** — prioritized action items
12. **Team Action Plan** — split by Bot Platform vs Capability Team
13. **Root Cause Analysis** — grouped priority view with tagging logic
14. **Data Quality** — methodology notes

---

## Root Cause Tags

Every escalated thread is automatically classified into one of these tags:

| Tag | Description |
|-----|-------------|
| `KB_GAP_PARTIAL` | Bot found some docs but couldn't answer follow-up questions |
| `PREFILTER_DIRECT_ESC` | Prefilter rule escalated before bot attempted an answer |
| `WRONG_CHANNEL` | Engineer redirected user to another team/channel |
| `ANSWER_INSUFFICIENT` | User explicitly marked bot answer as insufficient |
| `BOT_OVER_CLARIFIED` | Bot entered clarification loop instead of answering |
| `BOT_CHOSE_ESCALATE` | Bot reasoned to escalate despite having context |
| `USER_DIRECT_ESC` | User clicked escalate without receiving a bot answer |
| `NO_KB_COVERAGE` | Bot found no relevant documents in the knowledge base |
| `OTHER` | No signal matched |

---

## Key Metrics (March 2026 Baseline)

- **Bot resolution rate**: 22–29% (weeks 2–5)
- **Total threads**: 819 across 6 weeks (Feb 23 – Mar 30)
- **Escalations**: 663 of 819 threads (81%)
- **Prefiltered**: 227 threads — 202 (89%) directly escalated by prefilter rule
- **Target**: 35%+ resolution rate

---

## Tech Stack

- **Python 3.10+**
- [Streamlit](https://streamlit.io) — web app UI
- [pandas](https://pandas.pydata.org) — data processing
- [scikit-learn](https://scikit-learn.org) — TF-IDF + KMeans clustering
- [numpy](https://numpy.org) — numerical operations

---

## Team

| Area | Owner |
|------|-------|
| Analysis & Report | Dev Success Team |
| Bot Platform fixes | Bot Platform Team |
| KB Content fixes | Identity Capability Team |
| Routing fixes | Identity Team + Bot Platform |
