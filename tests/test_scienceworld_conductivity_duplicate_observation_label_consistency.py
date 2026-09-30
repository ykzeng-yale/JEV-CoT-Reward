import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_SPEC = importlib.util.spec_from_file_location(
    "alias_runner", ROOT / "scripts/run_scienceworld_conductivity_duplicate_observation_label_consistency.py"
)
RUNNER = importlib.util.module_from_spec(RUN_SPEC)
assert RUN_SPEC.loader is not None
RUN_SPEC.loader.exec_module(RUNNER)
SPEC = importlib.util.spec_from_file_location(
    "alias_audit", ROOT / "scripts/audit_scienceworld_conductivity_duplicate_observation_label_consistency.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_source_label_reads_only_named_task_modifier():
    class JavaClass:
        def __init__(self, name):
            self.name = name

        def getSimpleName(self):
            return self.name

    class Modifier:
        def __init__(self, name, key=None, value=None):
            self._class = JavaClass(name)
            self._key = key
            self._value = value

        def getClass(self):
            return self._class

        def key(self):
            if self._key is None:
                raise AssertionError("non-value modifiers must be skipped before key()")
            return self._key

        def value(self):
            return self._value

    class Task:
        def taskModifiers(self):
            return [
                Modifier("TaskObject"),
                Modifier("TaskValueBool", "unknownIsConductive", True),
            ]

    class Interface:
        def task(self):
            return Task()

    class Option:
        def get(self):
            return Interface()

    class Server:
        def agentInterface(self):
            return Option()

    class Env:
        server = Server()

    assert RUNNER.source_label(Env()) is True


def test_auditor_detects_cross_split_conflict_without_exported_label_values(tmp_path):
    protocol = {
        "protocol": "test",
        "upstream_result_sha256": "upstream",
        "source_members": {"a": {"sha256": "a"}, "b": {"sha256": "b"}},
        "splits": {
            "train": {"start_inclusive": 0, "end_exclusive": 300},
            "dev": {"start_inclusive": 300, "end_exclusive": 450},
            "test": {"start_inclusive": 450, "end_exclusive": 600},
        },
    }
    rows = []
    for i in range(300):
        rows.append({"split": "train", "variation_id": i, "observation_sha256": f"{i:064x}", "label_matches_lowest_id_same_observation": True})
    for i in range(300, 450):
        rows.append({"split": "dev", "variation_id": i, "observation_sha256": f"{i:064x}", "label_matches_lowest_id_same_observation": True})
    shared_hash = "f" * 64
    rows[0]["observation_sha256"] = shared_hash
    rows[300]["observation_sha256"] = shared_hash
    rows[300]["label_matches_lowest_id_same_observation"] = False
    result = {
        "protocol": "test",
        "protocol_sha256": MODULE.hashlib.sha256(json.dumps(protocol).encode()).hexdigest(),
        "upstream_result_sha256": "upstream",
        "source_member_sha256": {"a": "a", "b": "b"},
        "split_counts": {"train": 300, "dev": 150, "test_loaded": 0},
        "records": rows,
        "raw_observations_written": False,
        "answer_label_values_written": False,
        "source_defined_label_values_read": True,
        "test_ids_loaded": False,
        "gold_paths_requested": False,
        "score_reward_values_read": 0,
        "model_calls": 0,
        "jev_calls": 0,
        "network_calls": 0,
    }
    protocol_path = tmp_path / "protocol.json"
    result_path = tmp_path / "result.json"
    protocol_path.write_text(json.dumps(protocol))
    result_path.write_text(json.dumps(result))
    report = MODULE.audit(protocol_path, result_path)
    assert report["cross_split_shared_exact_input_groups"] == 1
    assert report["cross_split_label_conflict_groups"] == 1
