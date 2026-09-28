import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from diagnose_repair_controller import fit_predict


def test_constant_training_targets_ignore_unseen_words():
    y=np.array([[.25,.75],[.25,.75],[.25,.75]])
    p=fit_predict(['graph path','sum numbers','graph numbers'],y,['unseen secret words'])
    np.testing.assert_allclose(p,[[.25,.75]])
