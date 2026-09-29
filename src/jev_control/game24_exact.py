"""Independent exact-arithmetic terminal checker for Game-of-24 outputs.

This intentionally does not share code with the inspected interwhen monitor.
It accepts only integer leaves and the four permitted binary operators.
Formatting/LaTeX extraction belongs in a separately audited parser.
"""

import ast
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import operator


_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


@dataclass(frozen=True)
class Game24Check:
    valid: bool
    value: Fraction | None
    reason: str


class InvalidExpression(ValueError):
    pass


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return Fraction(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise InvalidExpression("division by zero")
        return _OPERATORS[type(node.op)](left, right)
    raise InvalidExpression("only integer leaves and binary +, -, *, / are allowed")


def _integer_leaves(node):
    if isinstance(node, ast.Expression):
        return _integer_leaves(node.body)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return [node.value]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        return _integer_leaves(node.left) + _integer_leaves(node.right)
    raise InvalidExpression("only integer leaves and binary +, -, *, / are allowed")


def check_game24(expression, numbers):
    """Check a Python-style expression using each integer exactly once.

    The input must not include ``= 24``, LaTeX delimiters, or prose. Returns a
    structured failure rather than evaluating unsupported syntax.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        expected = list(numbers)
        if any(type(number) is not int for number in expected):
            raise InvalidExpression("task numbers must be integers")
        if len(expected) != 4:
            raise InvalidExpression("Game of 24 requires exactly four task numbers")
        if Counter(_integer_leaves(tree)) != Counter(expected):
            return Game24Check(False, None, "input numbers must each be used exactly once")
        value = _evaluate(tree)
        if value != 24:
            return Game24Check(False, value, "expression does not equal 24 exactly")
        return Game24Check(True, value, "valid")
    except (SyntaxError, InvalidExpression, ZeroDivisionError, TypeError) as exc:
        return Game24Check(False, None, str(exc))
