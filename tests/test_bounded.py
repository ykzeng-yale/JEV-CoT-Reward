import importlib.util
from pathlib import Path
import sys
import pytest

spec=importlib.util.spec_from_file_location('bounded',Path(__file__).parents[1]/'scripts/run_bounded.py')
bounded=importlib.util.module_from_spec(spec);spec.loader.exec_module(bounded)


def test_success_and_refuse_reusing_control_directory(tmp_path):
    control=tmp_path/'run'
    result=bounded.run([sys.executable,'-c',"print('public result')"],control,
                       wall_seconds=3,rss_mib=512,output_mib=1,poll_seconds=.02)
    assert result['status']=='completed' and result['exit_code']==0
    assert (control/'stdout.log').read_text().strip()=='public result'
    with pytest.raises(FileExistsError):
        bounded.run([sys.executable,'-c',"raise Exception()"],control,
                    wall_seconds=3,rss_mib=512,output_mib=1)


def test_wall_timeout_retains_failure_receipt_and_reaps_own_worker(tmp_path):
    result=bounded.run([sys.executable,'-c','import time; time.sleep(20)'],tmp_path/'run',
                       wall_seconds=.08,rss_mib=512,output_mib=1,poll_seconds=.02)
    assert result['status']=='failed' and result['stop_reason']=='wall_limit'
    assert result['exit_code'] is not None
    assert result['elapsed_seconds']<5
