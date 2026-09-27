import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from diagnose_action_replicability import diagnose


def test_development_only_replicability_complete_disjoint_splits():
    root=Path(__file__).parents[1]/'artifacts/mechanism-v1-terminal'
    result=diagnose(root)
    assert result['problems']==24 and len(result['splits'])==6
    for split in result['splits']:
        assert set(split['selection_repeats']).isdisjoint(split['evaluation_repeats'])
        assert set(split['selection_repeats'])|set(split['evaluation_repeats'])==set(range(4))
    with pytest.raises(ValueError,match='development'):
        diagnose(root.parent/'prospective-v1-terminal')
