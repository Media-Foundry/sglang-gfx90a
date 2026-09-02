#!/usr/bin/env python3
"""Build deterministic long, heterogeneous DSV4 code-review prompts.

The workload uses real source from this checkout rather than repeated filler.
Each request asks a different engineering question and is encoded with the
checkpoint's official DeepSeek-V4 chat formatter.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from transformers import AutoTokenizer


FILES_AND_TASKS = (
    ("python/sglang/srt/server_args.py", "Audit configuration validation and identify one unsafe interaction."),
    ("python/sglang/srt/managers/scheduler.py", "Explain the scheduling state machine and propose a latency optimization."),
    ("python/sglang/srt/models/deepseek_v4.py", "Trace one DeepSeek-V4 forward path and flag a correctness hazard."),
    ("python/sglang/srt/mem_cache/memory_pool.py", "Review allocation invariants and design a focused regression test."),
    ("python/sglang/srt/model_loader/loader.py", "Review weight-loading concurrency and error propagation."),
    ("python/sglang/srt/layers/attention/dsa_backend.py", "Explain the sparse-attention dataflow and its likely bottleneck."),
    ("python/sglang/srt/models/deepseek_v2.py", "Compare the expert and attention dispatch boundaries in this model."),
    ("python/sglang/srt/managers/schedule_batch.py", "Find a batch-compaction edge case and suggest a test."),
    ("python/sglang/srt/utils/common.py", "Review distributed helper failure handling and propose a safer contract."),
    ("python/sglang/srt/managers/tokenizer_manager.py", "Identify avoidable host serialization in request admission."),
    ("python/sglang/srt/layers/attention/aiter_backend.py", "Explain backend selection and identify a ROCm-specific risk."),
    ("python/sglang/srt/layers/quantization/fp8.py", "Audit scale handling and name the most important numerical test."),
    ("python/sglang/srt/mem_cache/unified_radix_cache.py", "Explain eviction correctness under concurrent prefix reuse."),
    ("python/sglang/srt/distributed/parallel_state.py", "Review collective ordering and identify a deadlock scenario."),
    ("python/sglang/srt/arg_groups/overrides.py", "Simplify the override resolution rules without changing behavior."),
    ("python/sglang/srt/disaggregation/decode.py", "Trace decode handoff and identify one backpressure failure mode."),
    ("python/sglang/srt/mem_cache/multi_ended_allocator.py", "State the allocator invariants and construct an adversarial sequence."),
    ("python/sglang/srt/environ.py", "Audit experimental feature flags for ambiguous or unsafe defaults."),
    ("python/sglang/srt/entrypoints/openai/serving_chat.py", "Review OpenAI compatibility and identify an error-mapping gap."),
    ("python/sglang/srt/entrypoints/openai/serving_responses.py", "Trace streaming response assembly and test cancellation semantics."),
    ("python/sglang/kernels/ops/layernorm/mhc.py", "Explain the MHC numerical pipeline and its synchronization points."),
    ("python/sglang/srt/layers/moe/moe_runner/aiter.py", "Trace routed-expert preprocessing and identify redundant memory traffic."),
    ("python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py", "Review compressed-KV addressing at ring boundaries."),
    ("python/sglang/srt/managers/prefill_delayer.py", "Explain the delay policy and propose starvation safeguards."),
    ("python/sglang/kernels/ops/moe/gfx90a_fp4_expert_gemv.py", "Review JIT specialization and suggest a small-M CDNA2 tactic."),
    ("python/sglang/srt/managers/scheduler_components/request_receiver.py", "Audit cross-rank request ordering and shutdown behavior."),
    ("python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py", "Explain graph padding and identify a stale-metadata risk."),
    ("python/sglang/srt/layers/attention/deepseek_v4_backend.py", "Compare prefill and decode attention preparation."),
    ("python/sglang/srt/managers/io_struct.py", "Review serialization compatibility and propose a versioning strategy."),
    ("python/sglang/srt/sampling/sampling_batch_info.py", "Find per-step host work that could be made graph-static."),
    ("python/sglang/srt/model_executor/model_runner.py", "Trace a prefill batch from metadata preparation to model forward."),
    ("python/sglang/srt/mem_cache/base_prefix_cache.py", "Review cache ownership rules and propose a concurrency test."),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--model", type=Path, default=Path("/home/pc/models/modelscope"))
    parser.add_argument("--target-tokens", type=int, default=2304)
    parser.add_argument(
        "--request-count",
        type=int,
        choices=(1, 4, 8, 16, 32),
        default=32,
        help="Build only the first N heterogeneous requests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / ".agents/memory/dsv4_prefill_diverse_32_input_ids.json",
    )
    return parser.parse_args()


def load_encoder(model: Path):
    path = model / "encoding/encoding_dsv4.py"
    spec = importlib.util.spec_from_file_location("encoding_dsv4_local", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.encode_messages


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    encode_messages = load_encoder(args.model)
    requests = []
    for index, (relative, task) in enumerate(
        FILES_AND_TASKS[: args.request_count]
    ):
        source = (args.root / relative).read_text(errors="replace")

        def encode(chars: int) -> tuple[str, list[int]]:
            content = (
                f"You are reviewing `{relative}` from an inference server.\n"
                f"Task: {task}\n\nSource excerpt:\n```\n{source[:chars]}\n```"
            )
            prompt = encode_messages(
                [{"role": "user", "content": content}], thinking_mode="chat"
            )
            return prompt, tokenizer.encode(prompt, add_special_tokens=False)

        lo, hi = 0, len(source)
        best_prompt, best_ids = encode(0)
        while lo <= hi:
            mid = (lo + hi) // 2
            prompt, ids = encode(mid)
            if len(ids) <= args.target_tokens:
                best_prompt, best_ids = prompt, ids
                lo = mid + 1
            else:
                hi = mid - 1
        if len(best_ids) < args.target_tokens - 8:
            raise RuntimeError(
                f"{relative} produced only {len(best_ids)} tokens; source is too short"
            )
        requests.append(
            {
                "index": index,
                "source": relative,
                "task": task,
                "prompt_tokens": len(best_ids),
                "prompt_sha256": hashlib.sha256(best_prompt.encode()).hexdigest(),
                "input_ids": best_ids,
            }
        )
    encoded = json.dumps(
        {
            "format": "dsv4-prefill-diverse-code-v1",
            "model": str(args.model),
            "target_tokens": args.target_tokens,
            "requests": requests,
        },
        indent=2,
    ) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(
        f"wrote {args.output}: requests={len(requests)} "
        f"token_range={min(x['prompt_tokens'] for x in requests)}.."
        f"{max(x['prompt_tokens'] for x in requests)}"
    )


if __name__ == "__main__":
    main()
