#!/usr/bin/env python3
"""Replay selected real Q/KV bytes at different M, without loading a model."""

import argparse
import json
from pathlib import Path

import torch

from scripts.rocm.compare_dsv41_row_trace import row_metrics


def pack_scope(rows, capacity, batch, valid=None):
    # Captures are canonical per-token records; kernels consume page-planar.
    n = rows.shape[0]
    page = torch.cat((rows[:, :576].flatten(), rows[:, 576:].flatten())).cuda()
    page = page.view(1, n, 1, 584)
    ids = torch.full((batch, 1, capacity), -1, dtype=torch.int32, device="cuda")
    seq = torch.arange(n, dtype=torch.int32, device="cuda")
    if valid is not None:
        seq = seq.masked_fill(~valid.cuda(), -1)
    ids[:, 0, :n] = seq
    lengths = torch.full((batch,), n, dtype=torch.int32, device="cuda")
    return page, ids, lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from sglang.kernels.ops.attention.nsa_triton_decode import triton_fp8_attention_fwd
    import sglang.kernels.ops.attention.nsa_triton_decode.triton_mla_kernels_decode_fused as kernels
    from sglang.kernels.ops.attention.dsv4 import fused_rope_inplace
    from sglang.kernels.ops.attention.dsv4.wo_a_bf16_invariant import wo_a_bf16_invariant

    records = []
    for f in sorted(args.trace_dir.glob("*MQALayer*-call1-*.pt")):
        trace = torch.load(f, weights_only=True, map_location="cpu")
        c = trace["attention_contract"]
        initial = torch.load(str(f).replace("-call1-", "-call0-"), weights_only=True, map_location="cpu")
        w = initial["attention_contract"].get("wo_a_weight")
        absorb, accurate, invariant = [], [], []
        outputs = []
        for batch in (1, 204):
            q = c["q"].unsqueeze(1).repeat(batch, 1, 1, 1).cuda()
            main, ids, lens = pack_scope(c["swa_kv"], c["swa_topk_capacity"], batch)
            extra = extra_ids = extra_lens = None
            if "extra_kv" in c:
                extra, extra_ids, extra_lens = pack_scope(c["extra_kv"], c["extra_topk_capacity"], batch, c["extra_valid"])
            out, lse = triton_fp8_attention_fwd(
                q=q, k_cache=main, head_dim_v=512, softmax_scale=c["softmax_scale"],
                indices=ids, attn_sink=c["sink"].cuda(), extra_k_cache=extra,
                extra_indices_in_kvcache=extra_ids, topk_length=lens,
                extra_topk_length=extra_lens,
            )
            outputs.append(out[-1].cpu())
            if w is not None:
                # Preserve service stride: eight local heads are a view of
                # the 64-head attention output, not a contiguous repack.
                o = out[:, 0, :c["n_local_heads"], :]
                fused_rope_inplace(o[..., -64:], None, c["freqs_cis"].cuda(),
                                   positions=torch.zeros(batch, dtype=torch.int64, device="cuda"), inverse=True)
                x = o.reshape(batch, c["n_local_groups"], -1)
                weight = w.cuda()
                absorb.append(torch.einsum("tgd,grd->tgr", x, weight)[-1].cpu())
                accurate.append(torch.einsum("tgd,grd->tgr", x.float(), weight.float())[-1].bfloat16().cpu())
                inv = wo_a_bf16_invariant(x, weight)
                invariant.append(inv[-1].cpu())
                assert torch.equal(inv, inv[:1].expand_as(inv)), "candidate depends on row placement"
                if batch == 1:
                    reference = torch.einsum("tgd,grd->tgr", x.cpu().double(), w.double())[0].bfloat16()
            print("configuration", batch, getattr(kernels._fused_gather_attn_dsv4_dual_scope_kernel, "best_config", None), flush=True)
        result = {"layer": trace["layer_id"], "rank": trace["rank"],
                  "attention_M1_vs_M204": row_metrics(*outputs)}
        if w is not None:
            wob_num = trace["module_number"] - 1
            service_decode = torch.load(next(args.trace_dir.glob(f"rank{trace['rank']}-*RowParallelLinear-module{wob_num}-call1-*.pt")), weights_only=True, map_location="cpu")["args"][0]
            service_prefill = torch.load(next(args.trace_dir.glob(f"rank{trace['rank']}-*RowParallelLinear-module{wob_num}-call3-*.pt")), weights_only=True, map_location="cpu")["args"][0][-1:]
            result.update(woa_M1_vs_M204=row_metrics(*absorb),
                          reproduced_service_decode=row_metrics(absorb[0].reshape_as(service_decode), service_decode),
                          reproduced_service_prefill=row_metrics(absorb[1].reshape_as(service_prefill), service_prefill),
                          invariant_M1_vs_M204=row_metrics(*invariant),
                          invariant_vs_fp64=row_metrics(invariant[0], reference),
                          fp32_woa_M1_vs_M204=row_metrics(*accurate),
                          woa_M1_vs_fp64=row_metrics(absorb[0], reference),
                          woa_M204_vs_fp64=row_metrics(absorb[1], reference),
                          fp32_woa_M1_vs_fp64=row_metrics(accurate[0], reference),
                          fp32_woa_M204_vs_fp64=row_metrics(accurate[1], reference))
        print(json.dumps(result), flush=True)
        records.append(result)
    if not records:
        raise ValueError("No real attention contracts found")
    with args.output.open("x") as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    main()
