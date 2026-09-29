"""Fresh longer-horizon expression tasks; separate from the frozen six-number set."""
import random


def make_horizon_task(index: int, seed: int = 93027119):
    rng = random.Random(seed + index)
    x = [rng.randint(1, 9) for _ in range(8)]
    a, b, c, d, e, f, g, h = x
    target = (a + b) * c + d * e - f + g * h
    numbers = list(x)
    rng.shuffle(numbers)
    task_id = f'arithmetic_construction_h8-{seed}-{index}'
    prompt = (
        f'Use each of these eight numbers exactly once: {numbers}. Build an arithmetic expression equal to {target}. '
        'Allowed operations are +, -, *, / and parentheses; no concatenation, powers, extra constants, or unary minus. '
        'Explain intermediate reasoning briefly, checking alternatives. End with FINAL: followed by only the expression.'
    )
    return {'id': task_id, 'family': 'arithmetic_construction_h8', 'prompt': prompt,
            'data': {'numbers': numbers, 'target': target,
                     'witness': f'(({a}+{b})*{c})+({d}*{e})-{f}+({g}*{h})'}}
