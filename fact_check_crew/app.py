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
from show_audit import format_run
from supervisor import prepare, finish
from tools import build_tools, load_config

config = load_config(ROOT / "configs" / "config.yaml")
LOG_PATH = Path(os.environ.get("AUDIT_LOG_PATH", ROOT / "outputs" / "audit" / "audit_log.jsonl"))


@st.cache_resource(show_spinner="Loading models and connecting to Qdrant...")
def get_tools():
    return build_tools(config)


st.title("Fact-check crew with human approval")
st.write(
    "A Researcher, Writer and Critic answer from Wikipedia passages. "
    "You approve or reject the answer before it is released. Every step goes to an audit log."
)

try:
    llm, search_tool = get_tools()
except Exception as e:
    st.error(f"Setup failed: {e}")
    st.stop()

question = st.text_input("Question", value="What is baseball?")

if st.button("Run crew"):
    audit = AuditLog(LOG_PATH)
    st.session_state["audit"] = audit
    st.session_state["state"] = None
    st.session_state.pop("result", None)
    with st.spinner("Researcher, Writer and Critic are working..."):
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
    st.subheader("Answer waiting for your approval")
    st.write(state["draft"])
    st.write(f"Critic approved: **{verdict.approved}**. Revisions used: {state['n_revisions']}.")
    st.caption(f"Critic feedback: {verdict.feedback}")
    note = st.text_input("Note (optional)")
    col1, col2 = st.columns(2)
    if col1.button("Approve"):
        st.session_state["result"] = finish(audit, state, True, note)
        st.rerun()
    if col2.button("Reject"):
        st.session_state["result"] = finish(audit, state, False, note)
        st.rerun()

if result:
    st.subheader(f"Status: {result['status']}")
    if result["answer"]:
        st.write(result["answer"])
    else:
        st.write("No answer released.")

if audit:
    rows = [r for r in audit.read() if r["run_id"] == audit.run_id]
    with st.expander("Audit log for this run", expanded=bool(result)):
        st.code(format_run(rows, width=200), language=None)
