#!/usr/bin/env python3
"""Regression oracle: V4.1 rotates wq_b output without per-head Q RMSNorm."""

import argparse
import json
from pathlib import Path

import torch

from sglang.kernels.ops.attention.fused_qk_norm_rope_store import (
    fused_qk_norm_rope_swa_store,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(341)
    freq = 1.0 / (10000.0 ** (torch.arange(32, device="cuda").float() / 32))
    angles = torch.arange(512, device="cuda").float()[:, None] * freq
    cos, sin = angles.cos().contiguous(), angles.sin().contiguous()
    cases = []
    for m, heads in ((1, 8), (3, 8), (17, 8), (3, 16)):
        positions = (torch.arange(m, device="cuda", dtype=torch.int64) * 19) % 512
        q = (torch.randn(m, heads * 512, device="cuda") * 7).bfloat16()
        kv = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
        slots = torch.arange(m, device="cuda", dtype=torch.int32)

        def call(normalize_q):
            store = torch.zeros(m, 512, device="cuda", dtype=torch.bfloat16)
            kv_copy = kv.clone()
            kwargs = {} if normalize_q is None else {"normalize_q": normalize_q}
            out = fused_qk_norm_rope_swa_store(
                q, kv_copy, None, weight, 1e-6, 1e-6, 64, cos, sin,
                positions, swa_cache=store, swa_loc=slots, bf16_store=True,
                **kwargs,
            )
            return out, kv_copy, store

        candidate, new_kv, new_store = call(False)
        legacy, old_kv, old_store = call(None)
        explicit_legacy, _, _ = call(True)
        expected = q.view(m, heads, 512).float().clone()
        pairs = expected[..., -64:].reshape(m, heads, 32, 2).clone()
        c, s = cos[positions, None, :], sin[positions, None, :]
        rot = torch.stack((pairs[..., 0] * c - pairs[..., 1] * s,
                           pairs[..., 0] * s + pairs[..., 1] * c), dim=-1)
        expected[..., -64:] = rot.flatten(-2)
        err = candidate.float() - expected
        relative_l2 = (err.norm() / expected.norm()).item()
        row = {
            "m": m, "heads": heads,
            "non_rope_exact": torch.equal(candidate[..., :448], q.view(m, heads, 512)[..., :448]),
            "legacy_default_exact": torch.equal(legacy, explicit_legacy),
            "kv_exact": torch.equal(new_kv, old_kv) and torch.equal(new_store, old_store),
            "finite": bool(torch.isfinite(candidate).all()),
            "relative_l2": relative_l2,
            "old_vs_new_relative_l2": ((legacy.float() - candidate.float()).norm() / candidate.float().norm()).item(),
        }
        row["passed"] = all(row[k] for k in ("non_rope_exact", "legacy_default_exact", "kv_exact", "finite")) and relative_l2 < 0.003
        cases.append(row)
    report = {"cases": cases, "passed": all(row["passed"] for row in cases)}
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
