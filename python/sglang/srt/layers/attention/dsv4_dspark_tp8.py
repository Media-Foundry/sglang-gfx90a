"""Pure shape guard for the opt-in TP8 DSpark attention schedule."""


def m128_overlap_eligible(*, enabled, hip, tp_size, compress_ratio, rows,
                          batch_size, target_verify, width, unified_kv):
    return bool(
        enabled and hip and tp_size == 8 and compress_ratio == 4
        and rows == 128 and batch_size == 32 and target_verify
        and width == 4 and unified_kv
    )
