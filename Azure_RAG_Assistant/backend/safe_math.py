"""
Safe arithmetic evaluator.

The original version of this project used Python's built-in eval() to power
the agent's "Calculator" tool. That is a remote code execution vulnerability:
anyone chatting with the bot could make the LLM call the tool with a payload
like "__import__('os').system('...')" and run arbitrary code on the server.

This module only ever parses a Python AST and walks it, allowing nothing but
numbers and a fixed set of arithmetic operators. It cannot execute arbitrary
code, import anything, or call any function.
"""

import ast
import operator

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} is not allowed")
        return _ALLOWED_OPERATORS[op_type](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"Operator {op_type.__name__} is not allowed")
        return _ALLOWED_OPERATORS[op_type](_eval_node(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def safe_calculate(expression: str) -> str:
    """Evaluate a plain arithmetic expression like '55 * 3' safely."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"
