#!/usr/bin/env python3
"""Scenario arithmetic; never presented as a measured throughput guarantee."""
import argparse
import json
import math


def estimate(problems, checkpoints, actions, repeats, budget, prefix_tokens, tokens_per_second,
             judge_input_tokens=1200, price_per_million=.042):
    if any(x<=0 for x in (problems,checkpoints,actions,repeats,budget,tokens_per_second)):
        raise ValueError("Positive sizes and throughput required")
    if not 0<=prefix_tokens<budget:raise ValueError("Prefix must fit budget")
    continuations=problems*checkpoints*actions*repeats
    new_tokens=continuations*(budget-prefix_tokens)
    shared_tokens=problems*checkpoints*prefix_tokens
    queries=problems*checkpoints
    return {"kind":"upper_bound_token_scenario_not_measurement",
            "continuations":continuations,"continuation_output_token_ceiling":new_tokens,
            "shared_prefix_token_ceiling":shared_tokens,
            "decode_hours_at_input_throughput":(new_tokens+shared_tokens)/tokens_per_second/3600,
            "unique_checkpoint_jev_queries":queries,
            "jev_usd_at_input_token_assumption":queries*judge_input_tokens*price_per_million/1e6,
            "excludes":"Independent replication subset, test-policy episodes, local judge compute, model loading and any prefill overhead absent from input throughput"}


def main():
    p=argparse.ArgumentParser()
    for name,default in [("problems",48),("checkpoints",1),("actions",3),("repeats",4),("budget",1024),("prefix-tokens",128),("judge-input-tokens",1200)]:
        p.add_argument("--"+name,type=int,default=default)
    p.add_argument("--tokens-per-second",type=float,default=40)
    args=vars(p.parse_args())
    print(json.dumps(estimate(**args),indent=2))


if __name__=="__main__":main()
