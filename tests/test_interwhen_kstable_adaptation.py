import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_interwhen_kstable_adaptation import KStableDetector, extract_final_boxed, prompt
from audit_interwhen_kstable_taskset import is_solvable
from jev_control.game24_exact import check_game24


def test_kstable_repeated_equation_requires_all_task_numbers():
    d = KStableDetector([1, 1, 1, 8], k=2)
    assert d.stable("First: (1+1)*8\nThe expression is (1+1+1)*8\nThe expression is (1+1+1)*8\n")
    assert d.trigger_equation == "(1+1+1)*8"
    assert d.trigger_line_count == 3


def test_kstable_does_not_trigger_on_mismatched_numbers_or_nonarithmetic():
    d = KStableDetector([1, 2, 3, 8], k=2)
    assert not d.stable("(1+1+1)*8\n(1+1+1)*8\n")
    assert not d.stable("answer 1, 2, 3, 8\nanswer 1, 2, 3, 8\n")


def test_independent_terminal_expression_parser_and_checker():
    expr = extract_final_boxed("work <think>trace</think> \\boxed{(1+1+1)*8}")
    assert expr == "(1+1+1)*8"
    assert check_game24(expr, [1, 1, 1, 8]).valid
    assert not check_game24("2**3*3*1", [1, 2, 3, 3]).valid


def test_final_parser_requires_closed_thinking_block_when_opened():
    assert extract_final_boxed("<think>unfinished \\boxed{24}") is None


def test_prompt_contains_numbers_and_empty_box_only():
    text = prompt([2, 3, 4, 5])
    assert "2, 3, 4, 5" in text
    assert r"\boxed{}" in text


def test_exact_task_solver_uses_no_floating_point_heuristic():
    assert is_solvable([1, 1, 1, 8])
    assert not is_solvable([1, 1, 1, 1])
