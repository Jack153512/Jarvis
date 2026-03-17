"""
Safe math expression evaluation for quick responses.
Used to avoid LLM/web agent for simple arithmetic like "what is 5+5*162+90".
"""
from __future__ import annotations

import ast
import operator as op
import re
from typing import Optional, Tuple, Union

# Allowed operations (no eval, no imports, no exec)
_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type}")
        return _OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type}")
        return _OPERATORS[op_type](operand)
    raise ValueError(f"Unsupported node: {type(node)}")


def safe_eval_math(expr: str):
    """
    Safely evaluate a math expression (numbers, +, -, *, /, **, %, //, parentheses).
    Returns (result, None) on success, or (None, error_message) on failure.
    """
    if not expr or not isinstance(expr, str):
        return None, "Empty expression"
    expr = expr.strip()
    if not expr:
        return None, "Empty expression"
    # Only allow safe chars
    if not re.match(r"^[\d\s+\-*/().%]+$", expr):
        return None, "Invalid characters"
    try:
        tree = ast.parse(expr, mode="eval")
        if not isinstance(tree.body, (ast.BinOp, ast.UnaryOp, ast.Constant)):
            return None, "Not a simple expression"
        result = _eval_node(tree.body)
        if isinstance(result, (int, float)) and not (result != result or abs(result) == float("inf")):
            return result, None
        return None, "Invalid result"
    except SyntaxError as e:
        return None, f"Syntax error: {e}"
    except Exception as e:
        return None, str(e)


# Pattern to find math-like substrings (digits, operators, parens)
_MATH_EXTRACT = re.compile(r"([\d\s+\-*/().%xX×÷]+)")


def extract_math_expression(text: str) -> Optional[str]:
    """
    Extract a math expression from natural language.
    E.g. "what is 5+5x162+90" -> "5+5*162+90"
    Returns None if no clear math expression found.
    """
    if not text or len(text) > 200:
        return None
    t = text.strip()
    # Remove trailing punctuation and question words
    t = re.sub(r"[?.!]\s*$", "", t).strip()
    # Find all math-like substrings
    matches = _MATH_EXTRACT.findall(t)
    if not matches:
        return None
    # Prefer the longest match that has at least one operator and multiple tokens
    best = None
    for m in matches:
        cleaned = m.strip()
        cleaned = re.sub(r"[xX×]", "*", cleaned)
        cleaned = re.sub(r"÷", "/", cleaned)
        cleaned = re.sub(r"\s+", "", cleaned)
        if len(cleaned) >= 2 and any(c in cleaned for c in "+-*/"):
            if best is None or len(cleaned) > len(best):
                best = cleaned
    return best


def try_math_quick_response(text: str) -> Tuple[Optional[Union[float, int]], Optional[str]]:
    """
    If the input looks like a simple math question, evaluate and return the answer.
    Returns (result, formatted_response) on success, or (None, None) if not applicable.
    """
    expr = extract_math_expression(text)
    if not expr:
        return None, None
    result, err = safe_eval_math(expr)
    if err:
        return None, None
    # Format nicely (int if whole number)
    if isinstance(result, float) and result == int(result):
        result = int(result)
    return result, str(result)
