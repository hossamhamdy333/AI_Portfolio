import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from safe_math import safe_calculate


def test_basic_multiplication():
    assert safe_calculate("55 * 3") == "165"


def test_parentheses_and_division():
    assert safe_calculate("(12 + 8) / 4") == "5.0"


def test_rejects_arbitrary_code_execution():
    result = safe_calculate("__import__('os').system('echo pwned')")
    assert "Could not evaluate" in result


def test_rejects_attribute_access():
    result = safe_calculate("().__class__")
    assert "Could not evaluate" in result
