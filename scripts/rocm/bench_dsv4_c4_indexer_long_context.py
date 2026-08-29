#!/usr/bin/env python3
"""gfx90a DSV4 C4-indexer long-context scaling oracle (no model service)."""

import argparse
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("SGLANG_OPT_USE_TRITON_INDEXER_FULL", "1")

import torch

from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE,
    fp8_paged_mqa_logits_torch,
)


BATCH = 32
HEADS = 64
DIM = 128
PAGE = 64
TOPK = 512


def make_case(length: int):
    torch.manual_seed(1701 + length)
    device = torch.device("cuda")
    pages_per_req = (length + PAGE - 1) // PAGE
    total_pages = BATCH * pages_per_req
    q_bf16 = torch.randn(BATCH, 1, HEADS, DIM, device=device, dtype=torch.bfloat16)
    q = q_bf16.to(FP8_DTYPE)
    weights = torch.rand(BATCH, HEADS, device=device, dtype=torch.float32) + 0.25

    kv_values_bf16 = torch.randn(
        total_pages, PAGE, DIM, device=device, dtype=torch.bfloat16
    )
    kv_values = kv_values_bf16.to(FP8_DTYPE)
    scales = torch.rand(total_pages, PAGE, device=device, dtype=torch.float32) + 0.5
    # Production C4 pages use a structure-of-arrays byte layout despite the
    # public [page,64,1,132] view: 8192 FP8 value bytes, then 64 FP32 scales.
    packed = torch.empty(total_pages, PAGE * (DIM + 4), device=device, dtype=torch.uint8)
    packed[:, : PAGE * DIM].copy_(kv_values.view(torch.uint8).reshape(total_pages, -1))
    packed[:, PAGE * DIM :].copy_(scales.view(torch.uint8).reshape(total_pages, -1))
    kv = packed.view(FP8_DTYPE).view(total_pages, PAGE, 1, DIM + 4)

    page_table = torch.arange(total_pages, device=device, dtype=torch.int32).reshape(
        BATCH, pages_per_req
    )
    seq_lens = torch.full((BATCH,), length, device=device, dtype=torch.int32)
    return q, kv, weights, seq_lens, page_table, kv_values, scales


def logits(q, kv, weights, seq_lens, page_table, length):
    return fp8_paged_mqa_logits_torch(
        q, kv, weights, seq_lens, page_table, None, length, False
    )


def slot_convert(raw_indices, page_table, out):
    logical_page = raw_indices >> 6
    offset = raw_indices & 63
    physical_page = torch.gather(page_table, 1, logical_page.long())
    out.copy_(((physical_page << 6) | offset).to(torch.int32))
    return out


def capture(fn):
    for _ in range(3):
        held = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        held = fn()
    torch.cuda.synchronize()
    return graph, held


def time_graph(graph, inner):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(inner):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / inner


def torch_reference(q, kv_values, weights, scales, length):
    pages_per_req = (length + PAGE - 1) // PAGE
    keys = kv_values.reshape(BATCH, pages_per_req * PAGE, DIM)[:, :length].to(
        torch.bfloat16
    )
    query = q[:, 0].to(torch.bfloat16)
    score = torch.bmm(keys, query.transpose(1, 2)).float()
    score = score.relu_().mul_(weights[:, None, :]).sum(dim=2)
    scale = scales.reshape(BATCH, pages_per_req * PAGE)[:, :length]
    return score * scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="513,1024,4096,16384")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    records = []

    for length in lengths:
        q, kv, weights, seq_lens, page_table, kv_values, scales = make_case(length)
        score = logits(q, kv, weights, seq_lens, page_table, length)
        raw = torch.topk(score, TOPK, dim=1, largest=True, sorted=False).indices.to(
            torch.int32
        )
        slots = torch.empty(BATCH, TOPK, dtype=torch.int32, device="cuda")
        slot_convert(raw, page_table, slots)
        expected_slots = (
            torch.gather(page_table, 1, (raw >> 6).long()) * PAGE + (raw & 63)
        ).to(torch.int32)
        slot_exact = bool(torch.equal(slots, expected_slots))
        if not slot_exact:
            raise AssertionError("logical-to-physical slot conversion mismatch")
        torch.cuda.synchronize()

        ref = torch_reference(q, kv_values, weights, scales, length)
        diff = (score - ref).abs()
        denom = ref.abs().clamp_min(1e-5)
        ref_raw = torch.topk(ref, TOPK, dim=1, largest=True, sorted=False).indices
        exact_rows = int(
            (torch.sort(raw.long(), dim=1).values == torch.sort(ref_raw, dim=1).values)
            .all(dim=1)
            .sum()
        )
        overlaps = []
        for row in range(BATCH):
            overlaps.append(
                len(set(raw[row].cpu().tolist()) & set(ref_raw[row].cpu().tolist()))
            )

        slot_out = torch.empty_like(slots)
        graphs = {
            "logits": capture(lambda: logits(q, kv, weights, seq_lens, page_table, length))[0],
            "topk": capture(
                lambda: torch.topk(score, TOPK, dim=1, largest=True, sorted=False).indices
            )[0],
            "slot": capture(lambda: slot_convert(raw, page_table, slot_out))[0],
        }

        def full_fn():
            full_score = logits(q, kv, weights, seq_lens, page_table, length)
            full_raw = torch.topk(
                full_score, TOPK, dim=1, largest=True, sorted=False
            ).indices.to(torch.int32)
            full_slots = torch.empty(BATCH, TOPK, dtype=torch.int32, device="cuda")
            return full_score, full_raw, slot_convert(full_raw, page_table, full_slots)

        graphs["full"] = capture(full_fn)[0]
        samples = {name: [] for name in graphs}
        order_a = ["logits", "topk", "slot", "full"]
        order_b = list(reversed(order_a))
        for round_id in range(args.rounds):
            for name in order_a if round_id % 2 == 0 else order_b:
                samples[name].append(time_graph(graphs[name], args.inner))

        medians = {name: statistics.median(values) for name, values in samples.items()}
        records.append(
            {
                "length": length,
                "score_max_abs": float(diff.max()),
                "score_mean_abs": float(diff.mean()),
                "score_max_rel": float((diff / denom).max()),
                "topk_exact_rows": exact_rows,
                "topk_overlap_min": min(overlaps),
                "topk_overlap_mean": statistics.mean(overlaps),
                "slot_conversion_exact": slot_exact,
                "samples_us": samples,
                "median_us": medians,
                "full_fp32_logits_write_bytes": BATCH * length * 4,
                # Modeling a fused design where each CTA scans 4096 positions,
                # emits at most 512 (FP32 score, int32 index) candidates, then a
                # second kernel merges candidates and converts slots.
                "candidate_score_index_write_bytes_chunk4096": BATCH
                * ((length + 4095) // 4096)
                * TOPK
                * 8,
            }
        )
        print(json.dumps(records[-1], sort_keys=True), flush=True)
        del q, kv, weights, seq_lens, page_table, kv_values, scales, score, ref
        torch.cuda.empty_cache()

    result = {
        "format": "dsv4-c4-indexer-long-context-v1",
        "shape": {"batch": BATCH, "heads": HEADS, "dim": DIM, "page": PAGE, "topk": TOPK},
        "indexer_block_s": os.environ.get("SGLANG_DSV4_GFX90A_INDEXER_BLOCK_S", "default"),
        "records": records,
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
