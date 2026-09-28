from types import SimpleNamespace
import pytest
from jev_control.guard_runtime import scoped_generate

@pytest.mark.parametrize('fail',[False,True])
def test_sampling_and_context_restored_even_on_failure(fail):
    raw=SimpleNamespace(_sampler='original',top_p=.9)
    b=SimpleNamespace(backend=raw,context={'policy':'guard'})
    def generate(*args,**kwargs):
        assert raw._sampler==('sampler',1.5,.9)
        assert b.context=={'policy':'guard','temperature':1.5}
        if fail:raise RuntimeError('fixture')
        return 'result'
    b.generate=generate
    factory=lambda temp,top_p:('sampler',temp,top_p)
    if fail:
        with pytest.raises(RuntimeError):scoped_generate(b,[1],10,7,1.5,factory)
    else:assert scoped_generate(b,[1],10,7,1.5,factory)=='result'
    assert raw._sampler=='original' and b.context=={'policy':'guard'}
