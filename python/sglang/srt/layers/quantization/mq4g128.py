"""Experimental routed-expert-only MagnumQuant G128 utilities for gfx90a."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.nn import Module


GROUP_SIZE = 128
GROUP_BYTES = 72


def fwht128(x: torch.Tensor) -> torch.Tensor:
    """Normalized group-local FWHT-128 over the final dimension."""
    if x.shape[-1] % GROUP_SIZE:
        raise ValueError(f"MQ4G128 requires K % 128 == 0, got {x.shape[-1]}")
    original_shape = x.shape
    y = x.float().reshape(-1, GROUP_SIZE).contiguous()
    width = 1
    while width < GROUP_SIZE:
        y = y.reshape(-1, GROUP_SIZE // (2 * width), 2, width)
        left, right = y[:, :, 0], y[:, :, 1]
        y = torch.stack((left + right, left - right), dim=2).reshape(-1, GROUP_SIZE)
        width *= 2
    return (y * (1.0 / math.sqrt(GROUP_SIZE))).reshape(original_shape)


def quantize_mq4g128(weight: torch.Tensor) -> torch.Tensor:
    """Rotate K groups and encode affine INT4 as [..., K/128, 72] bytes."""
    if weight.ndim < 2:
        raise ValueError("MQ4G128 weight must have at least two dimensions")
    k = weight.shape[-1]
    rotated = fwht128(weight)
    groups = rotated.reshape(*weight.shape[:-1], k // GROUP_SIZE, GROUP_SIZE)
    lo = groups.amin(dim=-1)
    hi = groups.amax(dim=-1)
    scale = ((hi - lo) / 15.0).clamp_min(torch.finfo(torch.float32).tiny)
    q = torch.round((groups - lo.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp_(0, 15).to(torch.uint8)
    packed = q[..., 0::2] | (q[..., 1::2] << 4)
    out = torch.empty((*groups.shape[:-1], GROUP_BYTES), dtype=torch.uint8, device=weight.device)
    out[..., :4] = scale.contiguous().view(torch.uint8).reshape(*scale.shape, 4)
    out[..., 4:8] = lo.contiguous().view(torch.uint8).reshape(*lo.shape, 4)
    out[..., 8:] = packed
    return out.contiguous()


def dequantize_mq4g128(packed: torch.Tensor) -> torch.Tensor:
    if packed.shape[-1] != GROUP_BYTES or packed.dtype != torch.uint8:
        raise ValueError("MQ4G128 packed tensor must end in 72 uint8 bytes")
    scale = packed[..., :4].contiguous().view(torch.float32).squeeze(-1)
    zero = packed[..., 4:8].contiguous().view(torch.float32).squeeze(-1)
    nibbles = packed[..., 8:]
    q = torch.empty((*nibbles.shape[:-1], GROUP_SIZE), dtype=torch.uint8, device=packed.device)
    q[..., 0::2] = nibbles & 15
    q[..., 1::2] = nibbles >> 4
    return q.float() * scale.unsqueeze(-1) + zero.unsqueeze(-1)


def _dequant_checkpoint_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Expand Qwen's [128,128] block-FP8 tensor for one-time requantization."""
    if weight.ndim != 3 or scale.ndim != 3:
        raise ValueError(f"expected expert FP8 [E,N,K] + scales, got {weight.shape}, {scale.shape}")
    e, n, k = weight.shape
    if scale.shape != (e, (n + 127) // 128, (k + 127) // 128):
        raise ValueError(f"unexpected Qwen FP8 scale shape {scale.shape} for weight {weight.shape}")
    expanded = scale.repeat_interleave(128, 1).repeat_interleave(128, 2)[:, :n, :k]
    return weight.float() * expanded


def _requantize_checkpoint_fp8_mq4g128(
    weight: torch.Tensor, scale: torch.Tensor, expert_chunk: int = 4
) -> torch.Tensor:
    """Stream FP8 experts into MQ4G128 without a whole-layer FP32 expansion."""
    e, n, k = weight.shape
    packed = torch.empty(
        (e, n, k // GROUP_SIZE, GROUP_BYTES),
        dtype=torch.uint8,
        device=weight.device,
    )
    for begin in range(0, e, expert_chunk):
        end = min(e, begin + expert_chunk)
        chunk = _dequant_checkpoint_fp8(weight[begin:end], scale[begin:end])
        packed[begin:end].copy_(quantize_mq4g128(chunk))
        del chunk
    return packed


class Mq4g128RoutedMoEMethod:
    """Opt-in Qwen4Exp routed-expert MQ4G128 method for gfx90a.

    The checkpoint is still loaded through the normal FP8 method. Post-load,
    each layer is converted independently and its FP8 storage is released.
    """

    fuse_routed_scaling_factor_in_topk = False

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix

    def create_moe_runner(self, layer: Module, moe_runner_config) -> None:
        self.moe_runner_config = moe_runner_config
        if moe_runner_config.hidden_size != 2560 or moe_runner_config.intermediate_size_per_partition != 640:
            raise ValueError("Qwen4 MQ4G128 currently requires H=2560 and I=640")
        if moe_runner_config.top_k != 10 or moe_runner_config.activation != "silu":
            raise ValueError(
                f"Qwen4 MQ4G128 requires top_k=10 and silu, got "
                f"top_k={moe_runner_config.top_k}, activation={moe_runner_config.activation}"
            )
        if moe_runner_config.apply_router_weight_on_input:
            raise ValueError("MQ4G128 does not yet support router weights on gate/up input")

    def create_weights(self, layer: Module, *args, **kwargs) -> None:
        self._fp8.create_weights(layer, *args, **kwargs)

    def process_weights_after_loading(self, layer: Module) -> None:
        self._fp8.process_weights_after_loading(layer)
        if not self._fp8.block_quant:
            raise ValueError("Qwen4 MQ4G128 online conversion requires block-FP8 checkpoint weights")
        packed13 = _requantize_checkpoint_fp8_mq4g128(
            layer.w13_weight.data, layer.w13_weight_scale_inv.data
        )
        packed2 = _requantize_checkpoint_fp8_mq4g128(
            layer.w2_weight.data, layer.w2_weight_scale_inv.data
        )
        # Preserve Parameter identity: model weight loaders may retain the
        # object returned by named_parameters() through post-processing. A new
        # Parameter would keep the old FP8 storage alive until loader teardown.
        layer.w13_weight.data = packed13
        layer.w2_weight.data = packed2
        layer.w13_weight.requires_grad_(False)
        layer.w2_weight.requires_grad_(False)
        del layer.w13_weight_scale_inv
        del layer.w2_weight_scale_inv
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        layer._mq4g128_routed = True
        torch.cuda.empty_cache()

    def _project(self, x: torch.Tensor, weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        from sglang.kernels.ops.moe.gfx90a_mq4g128_moe import (
            mq4g128_grouped,
            mq4g128_indexed,
        )
        from sglang.srt.environ import envs

        x_rot = fwht128(x).contiguous()
        valid = ids >= 0
        valid_count = int(valid.sum().item())
        unique_count = int(torch.unique(ids[valid]).numel()) if valid_count else 0
        occupancy = valid_count / max(unique_count, 1)
        threshold = envs.SGLANG_QWEN4_GFX90A_MQ4G128_GROUPED_OCCUPANCY.get()
        if occupancy >= threshold and valid_count >= 4:
            return mq4g128_grouped(x_rot, weight, ids)
        return mq4g128_indexed(x_rot, weight, ids)

    def apply(self, layer: Module, dispatch_output):
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        topk = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk):
            raise ValueError(f"MQ4G128 requires standard top-k output, got {topk.format}")
        x = dispatch_output.hidden_states
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(f"MQ4G128 input must be floating point, got {x.dtype}")
        ids = topk.topk_ids.to(torch.int32).contiguous()
        gate_up = self._project(x.float().contiguous(), layer.w13_weight, ids)
        intermediate = F.silu(gate_up[..., :640]) * gate_up[..., 640:]
        flat_intermediate = intermediate.reshape(-1, 640).contiguous()
        flat_ids = ids.reshape(-1, 1).contiguous()
        down = self._project(flat_intermediate, layer.w2_weight, flat_ids)
        down = down.reshape(x.shape[0], ids.shape[1], 2560)
        output = (down * topk.topk_weights.float().unsqueeze(-1)).sum(dim=1)
        return StandardCombineInput(hidden_states=output.to(x.dtype))
