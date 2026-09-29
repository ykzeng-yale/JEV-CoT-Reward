#!/usr/bin/env python3
"""Planning approximation for a paired binary-outcome difference.

For each independent task, D = Y_A - Y_B is in {-1, 0, 1}. Given a
planning discordance probability p=P(|D|=1), anticipated difference delta,
two-sided type-I error alpha, and target power, use

    n = (z_(1-alpha/2)*sqrt(p) + z_power*sqrt(p-delta^2))^2 / delta^2.

This is a normal planning approximation, not a guarantee or final analysis.
Use independent tasks as the unit; branches/replays are not extra tasks.
"""

from __future__ import annotations

import argparse
import json
import math
from statistics import NormalDist


def paired_binary_sample_size(
    discordance: float,
    effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    primary_contrasts: int = 1,
) -> int:
    """Return independent-task n using Bonferroni alpha across primaries."""
    values = (discordance, effect, alpha, power)
    if any(not math.isfinite(x) for x in values):
        raise ValueError("all probability/effect inputs must be finite")
    if not 0 <= discordance <= 1:
        raise ValueError("discordance must be in [0, 1]")
    if not 0 < effect <= 1:
        raise ValueError("effect must be in (0, 1]")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be in (0, 1)")
    if primary_contrasts < 1:
        raise ValueError("primary_contrasts must be at least 1")
    if effect > discordance:
        raise ValueError("effect cannot exceed paired discordance")
    null_variance = discordance
    alternative_variance = discordance - effect**2
    if alternative_variance <= 0:
        raise ValueError("planning variance must be positive")

    alpha_per_contrast = alpha / primary_contrasts
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha_per_contrast / 2)
    z_power = normal.inv_cdf(power)
    return math.ceil(
        (z_alpha * math.sqrt(null_variance)
         + z_power * math.sqrt(alternative_variance)) ** 2 / effect**2
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discordance", type=float, required=True,
                        help="planning P(Y_A != Y_B) across independent tasks")
    parser.add_argument("--effect", type=float, required=True,
                        help="minimum absolute risk difference, e.g. 0.03")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--primary-contrasts", type=int, default=1)
    args = parser.parse_args()
    n = paired_binary_sample_size(
        args.discordance,
        args.effect,
        alpha=args.alpha,
        power=args.power,
        primary_contrasts=args.primary_contrasts,
    )
    print(json.dumps({
        "independent_tasks": n,
        "discordance": args.discordance,
        "minimum_effect": args.effect,
        "alpha": args.alpha,
        "power": args.power,
        "primary_contrasts": args.primary_contrasts,
        "multiplicity": "Bonferroni across primary contrasts",
        "method": "paired-normal planning approximation; not final inference",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
