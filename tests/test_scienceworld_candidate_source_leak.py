import json
from pathlib import Path

import importlib.util

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_scienceworld_candidate_source_leak.py"
spec = importlib.util.spec_from_file_location("candidate_source_leak", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
audit = module.audit


ROOT = Path(__file__).resolve().parents[1]


def test_pinned_candidate_source_contracts_are_classified_conservatively():
    result = audit(
        ROOT / "configs/scienceworld_candidate_source_leak_v1.json",
        ROOT / "runs/scienceworld-prefix-replay-v1-control/artifacts/inputs/scienceworld-source.tar.gz",
    )
    assert result["status"] == "STATIC_SOURCE_RISKS_CONFIRMED"
    candidates = result["candidates"]
    assert candidates["mendelian_genetics"]["fixed_phenotype_pair_count"] == 16
    assert candidates["melting_point"]["fixed_letter_mapping_count"] == 26
    assert candidates["conductivity"]["classification"] == "seed_reconstruction_risk_not_proven"
    assert result["test_ids_loaded"] is False
    assert result["model_calls"] == result["jev_calls"] == result["network_calls"] == 0


def test_saved_result_is_static_evidence_not_dynamic_observation_audit():
    result = json.loads((ROOT / "results/scienceworld_candidate_source_leak_v1.json").read_text())
    assert result["status"] == "STATIC_SOURCE_RISKS_CONFIRMED"
    assert result["candidates"]["melting_point"]["dynamic_observation_audit"] == "not_run"
    assert result["candidates"]["mendelian_genetics"]["dynamic_observation_audit"].startswith("not_completed")
    assert "not observation contents" in result["interpretation_limit"]
