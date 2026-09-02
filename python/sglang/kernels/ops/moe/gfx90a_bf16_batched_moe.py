from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args


@cache_once
def _jit_dequant(e: int, n: int, k: int, blocks: int):
    args = make_cpp_args(e, n, k, blocks)
    return load_jit(
        "gfx90a_fp4_to_bf16_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_bf16_dequant_oracle.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aFp4ToBf16Oracle<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_helpers(e: int, m: int, t: int, h: int, i: int, p: int, a: int, blocks: int):
    args = make_cpp_args(e, m, t, h, i, p, a, blocks)
    return load_jit(
        "gfx90a_bf16_batched_moe_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_bf16_batched_moe_oracle.cuh"],
        cuda_wrappers=[
            ("build_runs", f"sglang::Gfx90aBf16BatchedMoeOracle<{args}>::build_runs"),
            ("pack", f"sglang::Gfx90aBf16BatchedMoeOracle<{args}>::pack"),
            ("swiglu", f"sglang::Gfx90aBf16BatchedMoeOracle<{args}>::swiglu"),
            ("reduce", f"sglang::Gfx90aBf16BatchedMoeOracle<{args}>::reduce"),
        ],
        extra_cuda_cflags=["-O3"],
    )


_workspaces: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
_ck_weight_workspaces: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _workspace(device: torch.device, p: int) -> dict[str, torch.Tensor]:
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, p)
    if key not in _workspaces:
        e, h, i, t, m = 256, 4096, 512, 6, 16384
        expert_buffer = torch.empty((e, p, h), dtype=torch.bfloat16, device=device)
        _workspaces[key] = {
            "expert_buffer": expert_buffer,
            "gate_up": torch.empty((e, p, 2 * i), dtype=torch.bfloat16, device=device),
            "intermediate": torch.empty((e, p, i), dtype=torch.bfloat16, device=device),
            "weight": torch.empty((e, 2 * i, h), dtype=torch.bfloat16, device=device),
            "route_rows": torch.empty((m, t), dtype=torch.int32, device=device),
            "starts": torch.empty(e, dtype=torch.int32, device=device),
            "counts": torch.empty(e, dtype=torch.int32, device=device),
            "overflow": torch.zeros(1, dtype=torch.int32, device=device),
        }
    return _workspaces[key]


def gfx90a_bf16_batched_moe_m16384(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_experts: torch.Tensor,
    num_valid_ids: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    padded_expert_rows: int = 512,
    blocks: int = 1664,
    limit: float = 10.0,
) -> torch.Tensor:
    e, m, t, h, i, a = 256, 16384, 6, 4096, 512, 64
    if hidden.shape != (m, h) or topk_ids.shape != (m, t):
        raise ValueError("BF16 batched MoE is restricted to exact M16384/Top-6")
    if tuple(w13.shape) != (e, 2 * i, h // 2) or tuple(w2.shape) != (e, h, i // 2):
        raise ValueError("BF16 batched MoE requires the TP4 DSV4 raw FP4 layout")
    ws = _workspace(hidden.device, padded_expert_rows)
    helper = _jit_helpers(e, m, t, h, i, padded_expert_rows, a, blocks)
    ws["overflow"].zero_()
    helper.build_runs(
        sorted_experts,
        num_valid_ids,
        ws["starts"],
        ws["counts"],
        ws["overflow"],
    )
    torch._assert_async(
        ws["overflow"] == 0,
        f"M16384 routed expert occupancy exceeds padded row cap {padded_expert_rows}",
    )
    helper.pack(
        hidden,
        sorted_ids,
        ws["starts"],
        ws["counts"],
        ws["expert_buffer"],
        ws["route_rows"],
    )
    weight13 = ws["weight"]
    _jit_dequant(e, 2 * i, h, blocks).run(
        w13.view(torch.uint8),
        s13.view(torch.uint8).reshape(e, 2 * i, h // 32),
        weight13,
    )
    torch.bmm(
        ws["expert_buffer"],
        weight13.transpose(1, 2),
        out=ws["gate_up"],
    )
    helper.swiglu(ws["gate_up"], ws["intermediate"], float(limit))
    weight2 = ws["weight"].view(-1)[: e * h * i].view(e, h, i)
    _jit_dequant(e, h, i, blocks).run(
        w2.view(torch.uint8),
        s2.view(torch.uint8).reshape(e, h, i // 32),
        weight2,
    )
    torch.bmm(
        ws["intermediate"],
        weight2.transpose(1, 2),
        out=ws["expert_buffer"],
    )
    if out is None:
        out = torch.empty((m, h), dtype=torch.bfloat16, device=hidden.device)
    helper.reduce(
        ws["expert_buffer"],
        topk_ids,
        ws["route_rows"],
        topk_weights,
        out,
    )
    return out


def gfx90a_bf16_ck_moe(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    blocks: int = 1664,
) -> torch.Tensor:
    e, t, h, i = 256, 6, 4096, 512
    m = hidden.shape[0]
    if m < 8192 or m > 36864:
        raise ValueError("BF16 CK MoE oracle is restricted to 8192 <= M <= 36864")
    if hidden.shape != (m, h) or topk_ids.shape != (m, t):
        raise ValueError("BF16 CK MoE oracle requires H4096/Top-6")
    device_index = hidden.device.index
    key = device_index if device_index is not None else torch.cuda.current_device()
    workspace = _ck_weight_workspaces.get(key)
    if workspace is None:
        workspace = (
            torch.empty((e, 2 * i, h), dtype=torch.bfloat16, device=hidden.device),
            torch.empty((e, h, i), dtype=torch.bfloat16, device=hidden.device),
        )
        _ck_weight_workspaces[key] = workspace
    weight13, weight2 = workspace
    _jit_dequant(e, 2 * i, h, blocks).run(
        w13.view(torch.uint8),
        s13.view(torch.uint8).reshape(e, 2 * i, h // 32),
        weight13,
    )
    _jit_dequant(e, h, i, blocks).run(
        w2.view(torch.uint8),
        s2.view(torch.uint8).reshape(e, h, i // 32),
        weight2,
    )

    import aiter.fused_moe as aiter_fused_moe
    from aiter import ActivationType, QuantType

    # gfx90a uses the generic CK two-stage implementation.  AIter's dispatch
    # table lacks the architecture key even though the generated CK instances
    # support gfx90a, so install an empty one-stage set and fall through to CK.
    aiter_fused_moe.fused_moe_1stage_dict.setdefault("gfx90a", set())
    return aiter_fused_moe.fused_moe(
        hidden_states=hidden,
        w1=weight13,
        w2=weight2,
        topk_weight=topk_weights,
        topk_ids=topk_ids,
        activation=ActivationType.Dsv4Silu,
        quant_type=QuantType.No,
        dtype=torch.bfloat16,
        moe_out=out,
    )


def gfx90a_bf16_ck_moe_m16384(*args, **kwargs) -> torch.Tensor:
    """Compatibility wrapper for the original standalone oracle."""
    if args and args[0].shape[0] != 16384:
        raise ValueError("expected M16384")
    return gfx90a_bf16_ck_moe(*args, **kwargs)
