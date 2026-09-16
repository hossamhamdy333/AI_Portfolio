# Exposes the agent as an MCP server, so any MCP client (Claude Desktop,
# Claude Code, or the demo client in notebooks/05_mcp_demo.ipynb) can
# query it. This has to be a script, not a notebook -- an MCP server is
# a long-running process that talks over stdio, not something you run
# cell by cell.
#
# Run it locally (stdio, no token needed):
#   python mcp_server.py
#
# Run it as a remote server (gated by a shared token - see README):
#   MCP_TRANSPORT=streamable-http MCP_ACCESS_TOKEN=<a-real-secret> python mcp_server.py

import os

from mcp.server.mcpserver import MCPServer
from mcp.server.auth.provider import AccessToken

import config
import portfolio

MCP_ACCESS_TOKEN = os.environ.get("MCP_ACCESS_TOKEN", "")
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")


class StaticTokenVerifier:
    """Checks a client's bearer token against a single shared secret.

    This only matters for a remote deployment (MCP_TRANSPORT=streamable-http
    or sse) - a local stdio connection (Claude Desktop, Claude Code) is
    already gated by whoever can launch this process on this machine in
    the first place, so requiring a token there too would protect nothing
    and just adds a step. That's why this verifier is only wired into the
    server when MCP_ACCESS_TOKEN is actually set (see build_server below),
    not unconditionally.

    Deliberately a single static secret, not a real OAuth client/token
    lifecycle - the actual problem this solves is "don't let a stranger
    who finds the URL burn through my Gemini quota", not multi-tenant
    access control. A shared secret is the right-sized fix for that.
    """

    async def verify_token(self, token: str) -> AccessToken | None:
        if MCP_ACCESS_TOKEN and token == MCP_ACCESS_TOKEN:
            return AccessToken(token=token, client_id="codebase-insight-agent-client", scopes=[])
        return None


def build_server() -> MCPServer:
    if MCP_ACCESS_TOKEN:
        from mcp.server.auth.settings import AuthSettings

        public_url = os.environ.get("MCP_PUBLIC_URL", "http://localhost:8000")
        return MCPServer(
            "codebase-insight-agent",
            token_verifier=StaticTokenVerifier(),
            # issuer_url/resource_server_url are metadata this SDK's auth
            # layer expects to be set whenever a token_verifier is used -
            # they don't need to be a real, separately-running OAuth
            # server for a single static secret like this one; verify_token
            # above is what actually gates every request.
            # validate_token_resource=False: this verifier doesn't set or
            # check an audience/resource claim on the token (it's a single
            # shared secret, not per-resource issued tokens), so the SDK
            # shouldn't try to validate one that was never set.
            auth=AuthSettings(issuer_url=public_url, resource_server_url=public_url, validate_token_resource=False),
        )
    return MCPServer("codebase-insight-agent")


mcp = build_server()

indexes = None
router = None
agent = None


def setup():
    """Loads the already-provisioned indexes, once, the first time any
    tool is called. Deliberately load_all_indexes(), not
    build_all_indexes() - this REQUIRES notebooks/01_indexing.ipynb to
    have been run first (against a real QDRANT_URL), and raises a clear
    error if it hasn't. That's intentional: the notebook is what actually
    provisions the real, persistent index every other part of this
    project depends on, not an optional demo of something this function
    would happily do again anyway."""
    global indexes, router, agent
    if indexes is None:
        print("Loading indexes...")
        indexes = portfolio.load_all_indexes()
        router = portfolio.build_router()
        agent = portfolio.build_agent(indexes, router)
        print(f"Ready. {len(indexes)} project(s) loaded.")


@mcp.tool()
def ask_portfolio(question: str) -> str:
    """Ask a question about Hossam's AI/ML portfolio. Routes to
    whichever project(s) the question is about and returns a grounded
    answer."""
    setup()
    return portfolio.ask(agent, question)["answer"]


@mcp.tool()
def list_projects() -> str:
    """List the portfolio projects this agent can currently answer
    questions about."""
    setup()
    return "\n".join(f"- {name}" for name in indexes)


@mcp.tool()
def compare_projects(project_a: str, project_b: str, aspect: str) -> str:
    """Compare two portfolio projects on a specific aspect, e.g.
    'evaluation approach' or 'deployment method'."""
    setup()
    question = f"Compare {project_a} and {project_b} on: {aspect}. Address each project in turn."
    return portfolio.ask(agent, question)["answer"]


if __name__ == "__main__":
    mcp.run(transport=MCP_TRANSPORT)
