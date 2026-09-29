from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from paired_binary_power import paired_binary_sample_size


def test_conservative_three_point_design_matches_program_plan():
    assert paired_binary_sample_size(0.30, 0.03) == 2614


def test_conservative_five_point_design_matches_program_plan():
    assert paired_binary_sample_size(0.30, 0.05) == 940


def test_six_primary_contrasts_apply_bonferroni():
    assert paired_binary_sample_size(0.30, 0.03, primary_contrasts=6) == 4034


def test_validation_rejects_effect_larger_than_discordance():
    with pytest.raises(ValueError, match="cannot exceed paired discordance"):
        paired_binary_sample_size(0.10, 0.11)


def test_planning_variance_must_be_positive():
    with pytest.raises(ValueError, match="variance must be positive"):
        paired_binary_sample_size(1.0, 1.0)
