"""Pure shape guard for the opt-in TP8 DSpark attention schedule."""


def m128_overlap_eligible(*, enabled, hip, tp_size, compress_ratio, rows,
                          batch_size, target_verify, width, unified_kv):
    return bool(
        enabled and hip and tp_size == 8 and compress_ratio == 4
        and rows == 128 and batch_size == 32 and target_verify
        and width == 4 and unified_kv
    )


def m128_ck_eligible(*, enabled, gfx90a, tp_size, compress_ratio, rows,
                     batch_size, target_verify, width, inverse_rope,
                     dspark=False, allow_c4=False):
    # HCA remains the default experiment; CSA requires its independent opt-in.
    return bool(enabled and dspark and gfx90a and tp_size == 8
                and (compress_ratio == 128 or (allow_c4 and compress_ratio == 4))
                and rows == 128 and batch_size == 32 and target_verify
                and width == 4 and not inverse_rope)


def refined_probability_eligible(*, ck_eligible, compress_ratio, enabled):
    return bool(ck_eligible and compress_ratio == 4 and enabled)
