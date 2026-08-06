"""
Shared state definition + persistent memory.
SQLite instead of Postgres/Redis — free, local file, zero setup.
"""
import os
from typing import TypedDict, Optional
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


class AgentState(TypedDict):
    task: str
    plan: str
    research: str
    analysis: str
    report: str
    needs_human_review: bool
    review_note: Optional[str]


def get_checkpointer(sqlite_path: str) -> SqliteSaver:
    os.makedirs(os.path.dirname(sqlite_path) or ".", exist_ok=True)
    conn = sqlite3.connect(sqlite_path, check_same_thread=False)
    return SqliteSaver(conn)
