#!/usr/bin/env python3
"""Reuse the variable-M CK oracle at the TP8 local intermediate size.

All CLI options and reference math come from the existing TP4 oracle.
This changes only synthetic weight/scale shapes, not weight precision.
"""

import sys

import bench_dsv4_m4608_bf16_ck_variable_m as oracle


if __name__ == "__main__":
    oracle.I = 256
    if "--runtime-shuffled-scales" in sys.argv:
        from aiter.ops.shuffle import shuffle_scale_a16w4

        sys.argv.remove("--runtime-shuffled-scales")
        original = oracle.gfx90a_bf16_ck_moe

        def runtime_layout(x, ids, weights, w13, s13, w2, s2, **kwargs):
            # Include the producer transform here for correctness only; this
            # wrapper's timing is NOT production stage timing (load-time work).
            packed13 = shuffle_scale_a16w4(s13.view(-1, 128), 256, True)
            packed2 = shuffle_scale_a16w4(s2.view(-1, 8), 256, False)
            return original(
                x, ids, weights, w13, packed13, w2, packed2,
                scales_shuffled=True, **kwargs
            )

        oracle.gfx90a_bf16_ck_moe = runtime_layout
    oracle.main()
