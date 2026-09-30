#!/usr/bin/env python3
"""Analyze only a complete development panel bound to a passing replay audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from statistics import mean


POLICIES = {"measurement", "masked_measurement", "continue_prior", "random_measurement"}
ENDPOINTS = ("task_success", "classification_correct", "placement_correct")


def _percentile(sorted_values: list[float], probability: float) -> float:
    index = (len(sorted_values) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (index - low)


def exact_group_bootstrap(group_differences: list[float]) -> dict:
    """Enumerate all 6**6 equally weighted draws of six observed substances.

    This is a finite-panel cluster sensitivity calculation. Six curated groups
    do not establish a random population sample or justify a population CI.
    """
    if len(group_differences) != 6 or any(not math.isfinite(x) or not -1 <= x <= 1 for x in group_differences):
        raise ValueError("exact sensitivity requires six finite group differences")
    values = sorted(sum(group_differences[i] for i in indices) / 6
                    for indices in product(range(6), repeat=6))
    return {
        "method": "exact equal-substance cluster bootstrap descriptive sensitivity",
        "groups": 6, "resamples": 6**6, "central_percentiles": [2.5, 97.5],
        "percentile_method": "linear interpolation at (N-1)*p",
        "interval": [_percentile(values, 0.025), _percentile(values, 0.975)],
        "population_confidence_interval": False,
        "caveat": "Only six observed target substances; not a population confidence interval, confirmatory test, or independent-family transfer estimate.",
    }


def analyze(protocol_path: Path, result_path: Path, audit_path: Path) -> dict:
    audit_raw = audit_path.read_bytes()
    report = json.loads(audit_raw)
    if (report.get("passes") is not True or report.get("record_checks_pass") is not True
            or report.get("replay_performed") is not True
            or report.get("replay_episodes_verified") != 600
            or report.get("replay_census_rows_verified") != 450
            or report.get("replay_error") is not None
            or not isinstance(report.get("checks"), dict) or not report["checks"]
            or any(value is not True for value in report["checks"].values())):
        raise ValueError("complete passing independent census and outcome replay audit required")
    result_raw, protocol_raw = result_path.read_bytes(), protocol_path.read_bytes()
    result_hash = hashlib.sha256(result_raw).hexdigest()
    protocol_hash = hashlib.sha256(protocol_raw).hexdigest()
    if report.get("result_sha256") != result_hash or report.get("protocol_sha256") != protocol_hash:
        raise ValueError("result or protocol changed after independent audit")
    # Parse outcome rates only after the immutable-file and replay gates above.
    result, protocol = json.loads(result_raw), json.loads(protocol_raw)
    if result.get("protocol_sha256") != protocol_hash or result.get("protocol") != protocol["protocol"]:
        raise ValueError("result does not match frozen protocol")
    if (set(protocol["policies"]) != POLICIES or len(protocol["policies"]) != 4
            or any(type(result.get(key)) is not int or result[key] != 0
                   for key in ("test_loaded", "model_calls", "jev_calls"))
            or result.get("gold_paths_requested") is not False):
        raise ValueError("development-only non-model study contract violated")
    dev = protocol["splits"]["dev"]
    ids = list(range(dev["start_inclusive"], dev["end_exclusive"]))
    if len(ids) != 150:
        raise ValueError("frozen development panel must contain exactly 150 variations")
    episodes = result["episodes"]
    keyed = {(row["variation_id"], row["policy"]): row for row in episodes}
    if (len(episodes) != 600 or len(keyed) != 600
            or set(keyed) != {(i, p) for i in ids for p in POLICIES}
            or any(row["split"] != "dev" for row in episodes)):
        raise ValueError("complete paired 150-by-four panel required; do not drop failures")
    groups: dict[str, list[int]] = defaultdict(list)
    for variation in ids:
        targets = {keyed[variation, policy]["target_group"] for policy in POLICIES}
        if len(targets) != 1:
            raise ValueError("paired policies disagree on target-substance group")
        groups[next(iter(targets))].append(variation)
    if len(groups) != 6 or any(len(values) != 25 for values in groups.values()):
        raise ValueError("frozen panel must have six substance groups with 25 variations each")
    for row in episodes:
        if any(type(row["endpoint"][endpoint]) is not bool for endpoint in ENDPOINTS):
            raise TypeError("audited outcome labels must be actual booleans")
        if row["failure"] is not None and row["endpoint"]["task_success"]:
            raise ValueError("failed policy episode cannot be counted as task success")

    policy_summary = {}
    group_summary = {group: {"variations": len(group_ids), "policies": {}}
                     for group, group_ids in sorted(groups.items())}
    for policy in protocol["policies"]:
        rows = [keyed[variation, policy] for variation in ids]
        known_intrinsic = [row for row in rows if row["intrinsic_action_count"] is not None]
        policy_summary[policy] = {
            "assigned_variations": 150, "analyzed_variations": 150,
            "failures": sum(row["failure"] is not None for row in rows),
            "failure_types": dict(Counter(row["failure"] for row in rows if row["failure"] is not None)),
            "outcomes": {endpoint: {"successes": sum(row["endpoint"][endpoint] for row in rows),
                                    "denominator": 150,
                                    "rate": mean(row["endpoint"][endpoint] for row in rows)} for endpoint in ENDPOINTS},
            "execution_eligibility": {
                "definition": "Frozen input panel includes all 150; the following are execution diagnostics, never filtering criteria.",
                "answer_decisions_committed": sum(row["decision_action_index"] is not None for row in rows),
                "public_measurement_readings_recorded": sum(row["reading_sha256"] is not None for row in rows),
                "six_circuit_connections_completed": sum(row["circuit_actions"] == 6 for row in rows),
            },
            "interventions": {
                "episodes_with_circuit_actions": sum(row["circuit_actions"] > 0 for row in rows),
                "circuit_connection_actions": sum(row["circuit_actions"] for row in rows),
            },
            "costs": {
                "executed_actions_including_padding_and_failures": sum(row["actions"] for row in rows),
                "intrinsic_actions_known_total": sum(row["intrinsic_action_count"] for row in known_intrinsic),
                "intrinsic_count_known_episodes": len(known_intrinsic),
                "intrinsic_count_unknown_episodes": len(rows) - len(known_intrinsic),
                "padding_actions_among_known_episodes": sum(row["actions"] - row["intrinsic_action_count"] for row in known_intrinsic),
                "summed_episode_wall_seconds": sum(row["wall_seconds"] for row in rows),
                "mean_episode_wall_seconds": mean(row["wall_seconds"] for row in rows),
                "model_calls": 0, "jev_calls": 0,
                "note": "Policy execution wall time excludes calibration, simulator reset census, integration gates, independent replay, and scheduler idle/setup overhead. Use Slurm accounting for allocation costs.",
            },
        }
        for group, group_ids in sorted(groups.items()):
            group_rows = [keyed[variation, policy] for variation in group_ids]
            group_summary[group]["policies"][policy] = {
                endpoint: mean(row["endpoint"][endpoint] for row in group_rows) for endpoint in ENDPOINTS}

    contrasts = {}
    for baseline in ("masked_measurement", "random_measurement", "continue_prior"):
        name = "measurement-minus-" + baseline
        contrasts[name] = {"primary": baseline == "masked_measurement", "outcomes": {}}
        for endpoint in ENDPOINTS:
            differences = {variation: int(keyed[variation, "measurement"]["endpoint"][endpoint])
                           - int(keyed[variation, baseline]["endpoint"][endpoint]) for variation in ids}
            group_differences = [mean(differences[i] for i in group_ids)
                                 for _, group_ids in sorted(groups.items())]
            contrasts[name]["outcomes"][endpoint] = {
                "paired_variations": 150,
                "wins": sum(value == 1 for value in differences.values()),
                "losses": sum(value == -1 for value in differences.values()),
                "ties": sum(value == 0 for value in differences.values()),
                "mean_paired_difference": mean(differences.values()),
                "difference_percentage_points": 100 * mean(differences.values()),
                "six_substance_group_mean_differences": dict(zip(sorted(groups), group_differences)),
                "descriptive_group_sensitivity": exact_group_bootstrap(group_differences),
            }

    return {
        "analysis": "scienceworld_conductivity_action_development_panel",
        "protocol": protocol["protocol"], "protocol_sha256": protocol_hash,
        "result_sha256": result_hash, "independent_audit_sha256": hashlib.sha256(audit_raw).hexdigest(),
        "audited_episode_count": 600, "audited_input_census_rows": 450,
        "development_variations": 150, "target_substance_groups": 6,
        "policy_summary": policy_summary, "substance_groups": group_summary, "paired_contrasts": contrasts,
        "input_census": report.get("input_census"),
        "calibration": result["calibration"],
        "scheduler_costs": {"allocated_cpu_hours": None, "allocated_gpu_hours": None,
                            "status": "Requires terminal Slurm accounting; model_calls=0 does not itself establish allocation hours."},
        "claim_limit": "Audited development-panel action evidence only. Six observed substances; descriptive sensitivity is not a population CI. No Jev incremental-value, unseen-test, independent-family transfer, or novel-controller claim.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "result", "audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.protocol, args.result, args.audit)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"analysis": report["analysis"], "audited_episode_count": report["audited_episode_count"]}))


if __name__ == "__main__":
    main()
