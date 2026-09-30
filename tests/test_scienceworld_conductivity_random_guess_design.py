from scripts.audit_scienceworld_conductivity_random_guess_design import audit


def fixture(n=4):
    protocol = {"random_seed": 9, "policies": ["random_measurement"],
                "splits": {"dev": {"start_inclusive": 300, "end_exclusive": 300+n}}}
    records = [{"split": "dev", "variation_id": 300+i,
                "target_group": "unknown substance O",
                "full_sha256": f"{i+1:064x}",
                # These fields must not affect the allocation audit.
                "label_matches": False} for i in range(n)]
    return protocol, {"records": records, "calibration": {"private": "ignored"}}


def test_random_allocation_uses_frozen_input_hash_and_not_row_id():
    protocol, census = fixture()
    result = audit(protocol, census)
    assert result["checks"]["uses_variation_id"] is False
    assert result["checks"]["uses_endpoint_or_outcome_fields"] is False
    assert result["development_rows"] == 4


def test_duplicate_visible_inputs_receive_same_seeded_bit():
    protocol, census = fixture()
    census["records"][1]["full_sha256"] = census["records"][0]["full_sha256"]
    result = audit(protocol, census)
    assert result["distinct_full_visible_inputs"] == 3
    assert result["duplicate_full_input_rows"] == 1


def test_incomplete_or_reordered_census_is_rejected():
    protocol, census = fixture()
    census["records"].reverse()
    try:
        audit(protocol, census)
    except ValueError as exc:
        assert "coverage/order" in str(exc)
    else:
        raise AssertionError("bad census order was accepted")
