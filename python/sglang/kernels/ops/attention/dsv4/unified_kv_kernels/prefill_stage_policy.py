"""Pure scope predicate for the opt-in gfx90a prefill pipeline experiment."""


def use_stage1(*, enabled, gfx90a, model_type, native_extend, speculative,
               draft, tp, cp, dcp, pp, rows, heads, dim, bf16, capturing):
    return bool(
        enabled and gfx90a and model_type == "deepseek_v4"
        and native_extend and not speculative and not draft
        and tp == 8 and cp == dcp == pp == 1
        and 8192 <= rows <= 65536 and heads == 8 and dim == 512
        and bf16 and not capturing
    )
