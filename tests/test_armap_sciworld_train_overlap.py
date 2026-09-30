import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from audit_armap_sciworld_train_overlap import ALLOWED_OUTPUT_KEYS, audit_bytes


def protocol_for(payload: bytes) -> dict:
    return {
        "dataset": {
            "repo": "test/repo",
            "revision": "a" * 40,
            "file": "train.json",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "candidate": {"heldout_variation_range": [360, 479]},
        "scan": {
            "candidate_terms": ["plant", "dominant", "recessive"],
            "identifier_key_fragments": ["task", "variation", "seed", "index", "id"],
        },
    }


def test_aggregate_overlap_screen_reports_topic_and_explicit_ids_without_text():
    sentinel = "NEVER_EMIT_PRIVATE_TRAJECTORY_OR_LABEL"
    payload = json.dumps(
        [
            {
                "id": "opaque-1",
                "conversations": [{"value": f"A plant {sentinel} shows a dominant trait."}],
                "preference": 1,
                "variationIdx": 420,
            },
            {"id": "opaque-2", "conversations": [{"value": "A recessive example."}]},
        ]
    ).encode()

    result = audit_bytes(payload, protocol_for(payload))

    assert set(result) == ALLOWED_OUTPUT_KEYS
    assert result["record_count"] == 2
    assert result["candidate_term_record_counts"] == {
        "plant": 1,
        "dominant": 1,
        "recessive": 1,
    }
    assert result["explicit_heldout_variation_field_records"] == 1
    assert result["rows_with_explicit_heldout_variation_ids"] == 1
    assert sentinel not in json.dumps(result)
    assert "preference" not in result


def test_fails_closed_on_wrong_frozen_input_digest():
    payload = b"[]"
    protocol = protocol_for(payload)
    protocol["dataset"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="frozen dataset identity"):
        audit_bytes(payload, protocol)


def test_rejects_non_list_or_non_record_top_level_schema():
    for payload in (b"{}", b"[1]"):
        with pytest.raises(ValueError, match="top-level list-of-records"):
            audit_bytes(payload, protocol_for(payload))
