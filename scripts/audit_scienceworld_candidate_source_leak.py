#!/usr/bin/env python3
"""Check pinned source contracts for direct or seed-reconstructable task cues."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_sources(protocol_path: Path, archive: Path) -> tuple[dict, dict[str, str]]:
    protocol = json.loads(protocol_path.read_text())
    if sha(archive.read_bytes()) != protocol["source_archive_sha256"]:
        raise ValueError("pinned archive hash mismatch")
    sources = {}
    with tarfile.open(archive, "r:gz") as tf:
        for name, record in protocol["members"].items():
            data = tf.extractfile(record["path"]).read()
            if sha(data) != record["sha256"]:
                raise ValueError(f"pinned source hash mismatch: {name}")
            sources[name] = data.decode()
    return protocol, sources


def audit(protocol_path: Path, archive: Path) -> dict:
    protocol, s = load_sources(protocol_path, archive)
    mendel_required = (
        "valueDominant", "valueRecessive", "new TaskValueStr(key = \"domOrRec\", value = domOrRec)",
        "new TaskValueStr(key = \"traitValue\", value = specificTraitValue)",
        "if (domOrRec.get == GeneticTrait.DOMINANT)",
        'description = "Your task is to determine whether " + traitValue.get',
        "GoalFind(objectName = answerBoxDom.get", "GoalFind(objectName = answerBoxRec.get",
    )
    if any(term not in s["mendelian_task"] for term in mendel_required):
        raise ValueError("Mendelian generator no longer matches fixed-map/shared-label source contract")
    plant_traits = len(re.findall(r"new GeneticTrait\(TRAIT_[A-Z_]+, valueDominant\s*=", s["unknown_plant_traits"]))
    if plant_traits != 16:
        raise ValueError(f"expected 16 fixed phenotype pairs, found {plant_traits}")

    melt_required = (
        "precomputedMeltPoints(letterName)", "substance.propMaterial.get.meltingPoint = this.precomputedMeltPoints(letterName)",
        'this.name = "unknown substance " + letterName',
        "if (meltingPoint.get >= tempPoint.get)",
        'description = "Your task is to measure the melting point of " + objectName.get',
        'description += "If the melting point of " + objectName.get + " is above " + tempPoint.get + " degrees celsius, focus on the " + boxAbove.get',
        'description += "If the melting point of " + objectName.get + " is below " + tempPoint.get + " degrees celsius, focus on the " + boxBelow.get',
    )
    if any(term not in s["melting_task"] + s["unknown_substances"] for term in melt_required):
        raise ValueError("melting task no longer matches fixed-letter-map/prompt source contract")
    melt_cases = len(re.findall(r'case "[A-Z]" => return [-0-9.]+', s["unknown_substances"]))
    if melt_cases != 26:
        raise ValueError(f"expected 26 fixed letter/melting-point entries, found {melt_cases}")

    conductivity_required = (
        "val randDouble = Random.nextFloat()", "if (randDouble < 0.50)",
        "value = unknownSubstance.propMaterial.get.electricallyConductive",
    )
    if any(term not in s["conductivity_task"] + s["unknown_substances"] for term in conductivity_required):
        raise ValueError("conductivity task no longer matches seeded-random-label source contract")
    seed_required = (
        "Random.setSeed(variationIdx)",
        "taskMaker = new TaskMaker1()",
        "tp.get.setupCombination(variationIdx, universe, agent)",
    )
    if any(term not in s["python_interface"] + s["task_maker"] for term in seed_required):
        raise ValueError("variation-seeded task construction contract changed")
    order_required = (
        "m <- unknownSubstancesSorted", "j <- partToPower", "n <- answerBoxes",
    )
    if any(term not in s["conductivity_task"] for term in order_required):
        raise ValueError("conductivity combination ordering changed")

    return {
        "protocol": protocol["protocol"],
        "source_revision": protocol["source_revision"],
        "source_archive_sha256": protocol["source_archive_sha256"],
        "source_member_sha256": {name: rec["sha256"] for name, rec in protocol["members"].items()},
        "status": "STATIC_SOURCE_RISKS_CONFIRMED",
        "candidates": {
            "mendelian_genetics": {
                "source_path": "public fixed phenotype pair -> visible queried value, and same domOrRec -> goal box",
                "fixed_phenotype_pair_count": plant_traits,
                "classification": "direct_source_rule_if_task_description_is_controller_visible",
                "dynamic_observation_audit": "not_completed; two same-cause Slurm failures preserved; no third retry",
            },
            "melting_point": {
                "source_path": "public fixed unknown-letter melting point -> threshold comparison -> goal box; prompt includes letter and threshold",
                "fixed_letter_mapping_count": melt_cases,
                "classification": "direct_source_rule_if_task_description_is_controller_visible",
                "dynamic_observation_audit": "not_run",
            },
            "conductivity": {
                "source_path": "random label generated with Random.nextFloat during TaskMaker construction; PythonInterface seeds from variationIdx; task combo order is public",
                "classification": "seed_reconstruction_risk_not_proven",
                "visible_signature_sufficiency": "not tested",
            },
        },
        "test_ids_loaded": False,
        "labels_or_outcomes_read": False,
        "model_calls": 0,
        "jev_calls": 0,
        "network_calls": 0,
        "gold_paths_requested": False,
        "interpretation_limit": "Static source contracts only. They establish algorithmic paths in public code, not observation contents, pretrained-model accuracy, action efficacy, or use of any task split.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("refuse to overwrite existing result")
    result = audit(a.protocol, a.archive)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "candidates": result["candidates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
