"""
Proves the connection-token gate actually works: right token accepted,
wrong/missing token rejected, and the default (no token configured, the
common local-stdio case) stays exactly as simple as before - no token
required at all.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("GOOGLE_API_KEY", "test")


def test_correct_token_is_accepted():
    os.environ["MCP_ACCESS_TOKEN"] = "the-real-secret"
    import importlib
    import mcp_server
    importlib.reload(mcp_server)

    verifier = mcp_server.StaticTokenVerifier()
    result = asyncio.run(verifier.verify_token("the-real-secret"))
    assert result is not None
    assert result.client_id == "codebase-insight-agent-client"


def test_wrong_token_is_rejected():
    os.environ["MCP_ACCESS_TOKEN"] = "the-real-secret"
    import importlib
    import mcp_server
    importlib.reload(mcp_server)

    verifier = mcp_server.StaticTokenVerifier()
    result = asyncio.run(verifier.verify_token("a-guess"))
    assert result is None


def test_empty_token_is_rejected():
    os.environ["MCP_ACCESS_TOKEN"] = "the-real-secret"
    import importlib
    import mcp_server
    importlib.reload(mcp_server)

    verifier = mcp_server.StaticTokenVerifier()
    result = asyncio.run(verifier.verify_token(""))
    assert result is None


def test_no_token_configured_means_no_auth_layer_at_all():
    """The common case: running locally over stdio for Claude Desktop /
    Claude Code, nobody sets MCP_ACCESS_TOKEN, and the server builds with
    no auth wired in at all - not a verifier that happens to always
    fail, an entirely absent auth layer, so local use stays exactly as
    simple as it was before this file existed."""
    os.environ.pop("MCP_ACCESS_TOKEN", None)
    import importlib
    import mcp_server
    importlib.reload(mcp_server)

    assert mcp_server.mcp.settings.auth is None


def test_server_registers_all_three_tools_regardless_of_auth():
    os.environ.pop("MCP_ACCESS_TOKEN", None)
    import importlib
    import mcp_server
    importlib.reload(mcp_server)

    tool_names = {t.name for t in mcp_server.mcp._tool_manager.list_tools()}
    assert tool_names == {"ask_portfolio", "list_projects", "compare_projects"}
