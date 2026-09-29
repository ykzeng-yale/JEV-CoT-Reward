from fractions import Fraction

import pytest

from jev_control.game24_exact import check_game24


def test_valid_fractional_solution_uses_each_number_once():
    result = check_game24("8/(3-8/3)", [3, 3, 8, 8])
    assert result.valid
    assert result.value == Fraction(24)


@pytest.mark.parametrize(
    "expression,numbers",
    [
        ("2**3*3*1", [1, 2, 3, 3]),  # excluded by the task's operator rules
        ("2//1+3+8+11", [1, 2, 3, 8]),
        ("(1+2+3+4)", [1, 2, 3, 4]),
        ("(1+2+3+4).__class__", [1, 2, 3, 4]),
        ("True+1+2+21", [1, 2, 3, 4]),
        ("1/0+2+3+4", [1, 2, 3, 4]),
    ],
)
def test_rejects_unsupported_or_incorrect_expressions(expression, numbers):
    assert not check_game24(expression, numbers).valid


@pytest.mark.parametrize(
    "expression,numbers",
    [
        ("(1+2+3+18)", [1, 2, 3, 4]),  # substituted input number
        ("(1+2+3+4+14)", [1, 2, 3, 4]),  # extra leaf
        ("(1+2+3+18)", [1, 2, 3, 4]),  # sums to 24 but substitutes a task number
    ],
)
def test_enforces_task_number_multiset(expression, numbers):
    assert not check_game24(expression, numbers).valid


def test_rejects_non_integer_task_data():
    assert not check_game24("1+2+3+18", [1, 2, 3, 18.0]).valid


def test_requires_four_task_numbers():
    assert not check_game24("1+2+3+18", [1, 2, 3]).valid
