"""
Multi-agent graph: Planner -> Researcher -> [human review if flagged] -> Analyst -> Writer

Kept as a simple linear state machine, not a fully dynamic agent swarm — matches
"simple, no over-engineering" while still demonstrating real orchestration:
state persistence, a human-in-the-loop interrupt, and specialized nodes.
"""
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from agents.tools import web_search, retrieve, run_python


def build_graph(llm, qdrant_client, checkpointer):

    def planner_node(state: AgentState) -> AgentState:
        prompt = (f"Break this task into a short plan: what to research, and whether "
                  f"data analysis/code is needed. Task: {state['task']}\n"
                  f"Reply in 3-5 bullet points, plain text.")
        plan = llm.invoke(prompt).content
        return {**state, "plan": plan}

    def researcher_node(state: AgentState) -> AgentState:
        web_results = web_search(state["task"])
        doc_results = retrieve(qdrant_client, state["task"])
        combined = f"WEB RESULTS:\n{web_results}\n\nDOCUMENT RESULTS:\n{doc_results}"

        # Simple conflict heuristic, decided by the LLM itself — this is what
        # triggers the human-in-the-loop pause.
        check_prompt = (f"Do these two sources meaningfully conflict on facts/numbers? "
                         f"Reply with exactly YES or NO, nothing else.\n\n{combined[:3000]}")
        flag = llm.invoke(check_prompt).content.strip().upper()
        needs_review = flag.startswith("YES")
        return {
            **state,
            "research": combined,
            "needs_human_review": needs_review,
            "review_note": "Conflicting information detected between sources." if needs_review else None,
        }

    def analyst_node(state: AgentState) -> AgentState:
        prompt = (f"Given this research, decide if a short Python snippet would help "
                  f"(quick math/aggregation). If yes, write ONLY the code. "
                  f"If not needed, reply exactly: NO_CODE_NEEDED\n\n"
                  f"Research:\n{state['research'][:3000]}")
        code_reply = llm.invoke(prompt).content.strip()
        if "NO_CODE_NEEDED" in code_reply:
            return {**state, "analysis": "No code analysis needed."}
        cleaned = code_reply.replace("```python", "").replace("```", "").strip()
        result = run_python(cleaned)
        return {**state, "analysis": f"Code:\n{cleaned}\n\nOutput:\n{result}"}

    def writer_node(state: AgentState) -> AgentState:
        prompt = (f"Write a concise Markdown report answering the task below, using the "
                  f"research and analysis. Use headers.\n\n"
                  f"Task: {state['task']}\n\nPlan: {state['plan']}\n\n"
                  f"Research: {state['research'][:4000]}\n\nAnalysis: {state['analysis']}")
        report = llm.invoke(prompt).content
        return {**state, "report": report}

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", END)

    # Pauses execution before "analyst" so a human can inspect state when
    # researcher flagged conflicting sources. Resume with app.invoke(None, config).
    return graph.compile(checkpointer=checkpointer, interrupt_before=["analyst"])
