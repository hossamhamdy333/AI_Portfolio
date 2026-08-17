"""Researcher -> Writer -> Critic, with a plain Python one-revision retry
loop -- not a CrewAI Flow. Three agents and "try again once if rejected"
is a while-loop, not a graph; reaching for Flow machinery here would be
solving a problem this project doesn't have.

The retry loop (run_crew_with_revision) is split from the actual
CrewAI-calling code (_run_initial_pass, _run_revision_pass) on purpose --
same shape as rag_router's eval_routing.py -- so the loop's logic (does it
stop after max_revisions, does it pass feedback correctly) can be unit
tested by mocking those two functions, without needing a live LLM call to
verify the control flow is right.
"""

import threading

from pydantic import BaseModel

from crewai import Agent, Task, Crew, Process


def _kickoff_in_thread(crew, inputs):
    """Run crew.kickoff() in a fresh thread with no event loop of its own.

    Notebooks (Jupyter/Colab) always have an asyncio event loop running in
    the kernel. Newer CrewAI versions detect that and refuse to run
    kickoff() synchronously from inside it -- raising RuntimeError and
    pointing at kickoff_async() instead. Rather than turn this project's
    public sync API (used as-is by tests and by eval_factcheck.py's plain
    for-loop with checkpointing) into an async one, isolate the call in a
    thread that has no running loop, so CrewAI's check never trips. Works
    the same whether the caller is a notebook cell or a plain script.
    """
    result, error = {}, {}

    def _run():
        try:
            result["value"] = crew.kickoff(inputs=inputs)
        except Exception as e:
            error["value"] = e

    thread = threading.Thread(target=_run)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result["value"]


class CriticVerdict(BaseModel):
    approved: bool
    feedback: str


def build_agents(llm, search_tool):
    researcher = Agent(
        role="Researcher",
        goal="Find passages from the indexed Wikipedia articles that are relevant to the question",
        backstory=(
            "A research assistant who always checks sources before anyone writes an answer. "
            "Never answers from memory -- only reports what the search tool actually returns."
        ),
        tools=[search_tool],
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    writer = Agent(
        role="Writer",
        goal="Write a clear, direct answer to the question using only the researcher's findings",
        backstory="A writer who answers only from the material they're given, and says so plainly when it isn't enough to answer confidently.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    critic = Agent(
        role="Critic",
        goal="Check whether every claim in the draft is actually supported by the researcher's passages",
        backstory="A careful fact-checker who rejects a draft the moment it states something the sources don't back up.",
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )
    return researcher, writer, critic


def _build_critic_task(critic, context_tasks):
    return Task(
        description=(
            "Check the draft answer against the researcher's passages. "
            "Approve it only if every claim in the draft is actually supported by those passages. "
            "If anything in the draft isn't backed up by the sources, reject it and say exactly which claim is unsupported."
        ),
        expected_output="A verdict: approved (true/false) and feedback explaining why.",
        agent=critic,
        context=context_tasks,
        output_pydantic=CriticVerdict,
    )


def _run_initial_pass(llm, search_tool, question):
    """Researcher -> Writer -> Critic, first attempt. Returns
    (draft, verdict, research_task) -- research_task is kept so a revision
    pass can reuse its output as context without searching again.
    """
    researcher, writer, critic = build_agents(llm, search_tool)

    research_task = Task(
        description="Research the question: {question}\n\nUse the search tool to find relevant passages.",
        expected_output="A list of the relevant passages found, with their titles and domains.",
        agent=researcher,
    )
    write_task = Task(
        description="Write an answer to: {question}\n\nUse only the researcher's findings above.",
        expected_output="A direct answer to the question, written only from the research findings.",
        agent=writer,
        context=[research_task],
    )
    critic_task = _build_critic_task(critic, [write_task, research_task])

    crew = Crew(agents=[researcher, writer, critic], tasks=[research_task, write_task, critic_task], process=Process.sequential)
    _kickoff_in_thread(crew, {"question": question})

    return str(write_task.output), critic_task.output.pydantic, research_task, writer, critic


def _run_revision_pass(llm, question, feedback, research_task, writer, critic):
    """Writer -> Critic only, reusing the original research (the sources
    didn't change, only the draft needs to). Returns (draft, verdict).
    """
    write_task = Task(
        description=(
            f"Write an answer to: {{question}}\n\nUse only the researcher's findings above.\n\n"
            f"A fact-checker rejected your previous draft for this reason: {feedback}\n"
            f"Revise the answer so it no longer makes that claim, or drop the claim if the sources don't support it."
        ),
        expected_output="A revised answer that addresses the fact-checker's feedback.",
        agent=writer,
        context=[research_task],
    )
    critic_task = _build_critic_task(critic, [write_task, research_task])

    crew = Crew(agents=[writer, critic], tasks=[write_task, critic_task], process=Process.sequential)
    _kickoff_in_thread(crew, {"question": question})

    return str(write_task.output), critic_task.output.pydantic


def run_crew_with_revision(llm, search_tool, question, max_revisions=1):
    """Researcher -> Writer -> Critic; if rejected, Writer gets
    max_revisions more attempts with the Critic's feedback, then whatever
    the Critic says on the last attempt is final. No open-ended loop.

    Returns a dict: {"answer", "approved", "n_revisions", "final_feedback",
    "research_passages"} -- research_passages is what the crew's own
    Researcher actually retrieved, not assumed to match some other
    caller's separate search call. A caller that wants to judge this
    answer's faithfulness should judge it against research_passages, not
    a passage set fetched independently -- otherwise a mismatch between
    the two searches (different phrasing, different top_k) could make the
    comparison unfair without anyone noticing.
    """
    draft, verdict, research_task, writer, critic = _run_initial_pass(llm, search_tool, question)
    research_passages = str(research_task.output)

    n_revisions = 0
    while not verdict.approved and n_revisions < max_revisions:
        n_revisions += 1
        draft, verdict = _run_revision_pass(llm, question, verdict.feedback, research_task, writer, critic)

    return {
        "answer": draft,
        "approved": verdict.approved,
        "n_revisions": n_revisions,
        "final_feedback": verdict.feedback,
        "research_passages": research_passages,
    }
