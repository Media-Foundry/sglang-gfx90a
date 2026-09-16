"""CPU-only prototype; not imported by the running service.

None is the existing MHC no-singleton-optimization hint, not a fabricated
scheduler batch size. The caller must supply the existing strict original-V4,
TP8, native eager-prefill mix-pair scope (including no graph/CP/TBO/draft).
"""


def resolve_batch_hint(batch_size, rows, *, enabled, active_scope,
                       bf16_dot=False, mfma=False, config_iters=False,
                       comb_refine=False):
    if not enabled or not active_scope or not 8192 <= rows <= 65536:
        return batch_size
    conflicts = [name for name, value in (
        ('BF16 MHC dot', bf16_dot), ('MFMA pre-mix', mfma),
        ('custom Sinkhorn iterations', config_iters),
        ('comb refinement', comb_refine),
    ) if value]
    if conflicts:
        raise ValueError('Common FP32/20 prefill conflicts with ' + ', '.join(conflicts))
    return None
