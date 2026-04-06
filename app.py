import streamlit as st
import sys
import os
import tempfile

st.set_page_config(
    page_title="Identity Bot Analyzer",
    page_icon="📊",
    layout="centered"
)

# Add data dir to path so we can import analyze_v4
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data'))
from analyze_v4 import run_analysis

# ── Header ───────────────────────────────────────────────────
st.title("📊 Identity Bot Analyzer")
st.markdown(
    "Upload a support bot CSV export to generate an interactive performance report. "
    "The report includes weekly trends, escalation root causes, cluster analysis, "
    "and team-specific recommendations."
)

# ── File Upload ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"],
    help="Export from your support bot platform. Must include columns: "
         "escalated, intent_category, conversation_history, ai_orchestrator_trace"
)

if uploaded_file:
    file_size_kb = len(uploaded_file.getvalue()) / 1024
    st.success(f"✅ Loaded: **{uploaded_file.name}** ({file_size_kb:.0f} KB)")

    if st.button("🚀 Generate Report", type="primary", use_container_width=True):

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(pct, msg):
                progress_bar.progress(pct)
                status_text.text(f"⏳ {msg}")

            html_content = run_analysis(tmp_path, progress_callback=update_progress)

            progress_bar.progress(1.0)
            status_text.text("✅ Report ready!")

            report_name = uploaded_file.name.replace(".csv", "_report.html")
            st.download_button(
                label="⬇️ Download Report",
                data=html_content.encode("utf-8"),
                file_name=report_name,
                mime="text/html",
                use_container_width=True,
                type="primary"
            )

            st.info(
                f"📄 Report size: {len(html_content)//1024} KB. "
                "Open the downloaded file in any browser."
            )

        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.exception(e)

        finally:
            os.unlink(tmp_path)

# ── Footer / Help ────────────────────────────────────────────
with st.expander("ℹ️ Expected CSV columns"):
    st.markdown("""
| Column | Required | Description |
|--------|----------|-------------|
| `escalated` | ✅ | TRUE/FALSE |
| `intent_category` | ✅ | howTo, troubleshooting, accessRequest, notification |
| `resolution_type` | ✅ | BOT or HUMAN |
| `conversation_history` | ✅ | JSON string |
| `ai_orchestrator_trace` | ✅ | JSON string |
| `week_start` | ✅ | Date column for week assignment |
| `faq_rendered` | Optional | TRUE/FALSE |
| `prefiltered` | Optional | TRUE/FALSE |
| `rca_human_available` | Optional | Survey fill flag |
    """)
