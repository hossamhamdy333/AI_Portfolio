import crew as crew_module


def prepare(llm, search_tool, question, audit, max_revisions=1):
    audit.log("start", question=question, max_revisions=max_revisions)

    draft, verdict, research_task, writer, critic = crew_module._run_initial_pass(llm, search_tool, question)
    audit.log("research", passages=str(research_task.output))
    audit.log("write", attempt=0, draft=draft)
    audit.log("critique", attempt=0, approved=verdict.approved, feedback=verdict.feedback)

    n_revisions = 0
    while True:
        if verdict.approved or n_revisions >= max_revisions:
            audit.log("decision", action="send_to_human", critic_approved=verdict.approved)
            break
        n_revisions += 1
        audit.log("decision", action="revise", reason=verdict.feedback)
        draft, verdict = crew_module._run_revision_pass(
            llm, question, verdict.feedback, research_task, writer, critic
        )
        audit.log("write", attempt=n_revisions, draft=draft)
        audit.log("critique", attempt=n_revisions, approved=verdict.approved, feedback=verdict.feedback)

    audit.log("approval_requested", answer=draft, critic_approved=verdict.approved, critic_feedback=verdict.feedback)
    return {"draft": draft, "verdict": verdict, "n_revisions": n_revisions}


def finish(audit, state, human_ok, note=""):
    audit.log("human_decision", approved=human_ok, note=note)
    status = "approved" if human_ok else "rejected"
    audit.log("final", status=status)
    return {
        "status": status,
        "answer": state["draft"] if human_ok else None,
        "critic_approved": state["verdict"].approved,
        "n_revisions": state["n_revisions"],
        "run_id": audit.run_id,
    }


def run_supervised(llm, search_tool, question, audit, approve, max_revisions=1):
    state = prepare(llm, search_tool, question, audit, max_revisions)
    human_ok, note = approve(state["draft"], state["verdict"])
    return finish(audit, state, human_ok, note)
