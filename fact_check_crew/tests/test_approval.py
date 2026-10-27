from types import SimpleNamespace

from approval import cli_approval


def fake_input(answers):
    it = iter(answers)
    return lambda prompt="": next(it)


def make_verdict():
    return SimpleNamespace(approved=True, feedback="fine")


def test_y_approves(monkeypatch):
    monkeypatch.setattr("builtins.input", fake_input(["y", "looks good"]))
    assert cli_approval("answer", make_verdict()) == (True, "looks good")


def test_n_rejects(monkeypatch):
    monkeypatch.setattr("builtins.input", fake_input(["N", ""]))
    assert cli_approval("answer", make_verdict()) == (False, "")


def test_bad_input_asks_again(monkeypatch):
    monkeypatch.setattr("builtins.input", fake_input(["maybe", "", "y", ""]))
    assert cli_approval("answer", make_verdict()) == (True, "")
