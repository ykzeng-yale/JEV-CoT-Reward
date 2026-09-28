"""Temperature-scoped bridge to the existing logged MLX backend."""


def scoped_generate(backend,prefix,cap,seed,temperature,make_sampler=None):
    if getattr(backend.backend,'quantization_config',{}).get('mode')=='unquantized_bfloat16':
        raw=backend.backend;previous=raw._sampler;context=dict(backend.context)
        try:
            raw._sampler=temperature;backend.context={**context,'temperature':temperature}
            return backend.generate(prefix,cap,seed,timeout_s=180)
        finally:
            raw._sampler=previous;backend.context=context
    if make_sampler is None:
        from mlx_lm.sample_utils import make_sampler
    raw=backend.backend;previous=raw._sampler
    context=dict(backend.context)
    try:
        raw._sampler=make_sampler(temp=temperature,top_p=raw.top_p)
        backend.context={**context,'temperature':temperature}
        return backend.generate(prefix,cap,seed,timeout_s=180)
    finally:
        raw._sampler=previous
        backend.context=context
