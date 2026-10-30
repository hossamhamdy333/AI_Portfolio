import json
import os
import sys
from pathlib import Path

os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from audit import AuditLog
from supervisor import prepare, finish
from tools import build_tools, load_config

st.set_page_config(page_title="Fact-check crew", layout="centered")

config = load_config(ROOT / "configs" / "config.yaml")
LOG_PATH = Path(os.environ.get("AUDIT_LOG_PATH", ROOT / "outputs" / "audit" / "audit_log.jsonl"))
MAX_RUNS = 5
REPO_URL = "https://github.com/hossamhamdy333/AI_Portfolio/tree/main/fact_check_crew"


@st.cache_resource(show_spinner="Loading models and connecting to Qdrant...")
def get_tools():
    return build_tools(config)


def detail(row):
    skip = ("run_id", "time", "step")
    parts = [f"{k}: {str(v)[:100]}" for k, v in row.items() if k not in skip]
    return " | ".join(parts).replace("\n", " ")


with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. The Researcher searches a sample of Wikipedia articles.\n"
        "2. The Writer drafts an answer from those passages only.\n"
        "3. The Critic checks the draft against the passages.\n"
        "4. You approve or reject the answer. Every step is logged."
    )
    st.caption("Topics: sports, technology, history and English literature.")
    st.markdown(f"[Source code]({REPO_URL})")

st.title("Fact-check crew")
st.caption("Three agents answer from Wikipedia passages. Nothing is released until you approve it.")

try:
    llm, search_tool = get_tools()
except Exception as e:
    st.error(f"Setup failed: {e}")
    st.stop()

with st.form("ask"):
    question = st.text_input(
        "Question",
        key="question",
        placeholder="Ask about sports, technology, history or English literature",
    )
    submitted = st.form_submit_button("Run crew")

if submitted:
    if not question.strip():
        st.warning("Type a question first.")
    elif st.session_state.get("runs", 0) >= MAX_RUNS:
        st.warning("Run limit reached for this session.")
    else:
        audit = AuditLog(LOG_PATH)
        st.session_state["audit"] = audit
        st.session_state["state"] = None
        st.session_state.pop("result", None)
        st.session_state["runs"] = st.session_state.get("runs", 0) + 1
        with st.spinner("The crew is working."):
            try:
                st.session_state["state"] = prepare(
                    llm, search_tool, question, audit, config["revision"]["max_revisions"]
                )
            except Exception as e:
                audit.log("error", message=str(e))
                st.error(f"The crew failed: {e}")

state = st.session_state.get("state")
audit = st.session_state.get("audit")
result = st.session_state.get("result")

if state and not result:
    verdict = state["verdict"]
    st.subheader("Review the answer")
    st.write(state["draft"])
    if verdict.approved:
        st.success("Critic: approved")
    else:
        st.warning("Critic: not approved")
    st.caption(f"{verdict.feedback} Revisions used: {state['n_revisions']}.")

    st.text_input("Note (optional)", key="note")
    col1, col2 = st.columns(2)
    if col1.button("Approve and release", type="primary", width="stretch"):
        st.session_state["result"] = finish(audit, state, True, st.session_state["note"])
        st.rerun()
    if col2.button("Reject", width="stretch"):
        st.session_state["result"] = finish(audit, state, False, st.session_state["note"])
        st.rerun()

if result:
    if result["answer"]:
        st.success("Approved. Answer released:")
        st.write(result["answer"])
    else:
        st.info("Rejected. No answer was released.")

if audit:
    rows = [r for r in audit.read() if r["run_id"] == audit.run_id]

    research = next((r["passages"] for r in rows if r["step"] == "research"), None)
    if research:
        with st.expander("Sources found by the Researcher"):
            st.markdown(research)

    with st.expander("Audit log for this run", expanded=bool(result)):
        table = [{"Time": r["time"][11:], "Step": r["step"], "Detail": detail(r)} for r in rows]
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button(
            "Download as JSONL",
            data="\n".join(json.dumps(r) for r in rows),
            file_name=f"audit_{audit.run_id}.jsonl",
            mime="application/json",
            on_click="ignore",
        )
