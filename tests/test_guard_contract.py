import math
import pytest
from jev_control.guard_contract import entropy_trigger,branch_score,BRANCH_TEMPERATURES


def test_current_inclusive_quantile_delays_first_possible_trigger():
    for n in range(1,11):
        assert not entropy_trigger(list(range(n)),1000)
    assert entropy_trigger(list(range(11)),1000)
    assert not entropy_trigger(list(range(11)),200)
    assert not entropy_trigger([1.]*11,1000)


def test_marker_bonus_is_not_correctness_and_empty_entropy_cannot_win():
    assert branch_score(2.,[1.,1.])==1.
    assert branch_score(2.,[3.,3.],True)==9.
    assert branch_score(2.,[],True)==-math.inf
    assert BRANCH_TEMPERATURES==(0.,.6,1.5)
    with pytest.raises(ValueError):entropy_trigger([float('nan')],1000)
