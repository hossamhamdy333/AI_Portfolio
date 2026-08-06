from agents.tools import run_python, web_search


def test_run_python_prints_output():
    result = run_python("print(2 + 2)")
    assert result.strip() == "4"


def test_run_python_no_output_message():
    result = run_python("x = 1 + 1")
    assert result == "Code ran with no printed output."


def test_run_python_catches_errors_safely():
    result = run_python("1 / 0")
    assert result.startswith("Error:")


def test_run_python_blocks_unsafe_builtins():
    # __import__ / open aren't in SAFE_BUILTINS, so this should fail cleanly,
    # not actually touch the filesystem.
    result = run_python("open('/etc/passwd').read()")
    assert result.startswith("Error:")


def test_run_python_allows_pandas_numpy():
    result = run_python("import pandas as pd\nprint(pd.Series([1,2,3]).sum())")
    assert result.strip() == "6"


def test_web_search_handles_failure_gracefully(monkeypatch):
    class BoomDDGS:
        def __enter__(self):
            raise RuntimeError("rate limited")

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("agents.tools.DDGS", BoomDDGS)
    result = web_search("anything")
    assert "unavailable" in result.lower()


def test_web_search_no_results(monkeypatch):
    class EmptyDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def text(self, query, max_results=5):
            return []

    monkeypatch.setattr("agents.tools.DDGS", EmptyDDGS)
    assert web_search("anything") == "No results found."


def test_web_search_formats_results(monkeypatch):
    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def text(self, query, max_results=5):
            return [{"title": "T", "body": "B", "href": "http://x.com"}]

    monkeypatch.setattr("agents.tools.DDGS", FakeDDGS)
    result = web_search("anything")
    assert "T" in result and "B" in result and "http://x.com" in result
