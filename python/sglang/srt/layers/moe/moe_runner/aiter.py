from __future__ import annotations

import functools
import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import get_bool_env_var, get_int_env_var, is_gfx95_supported

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.base import CombineInput
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLDispatchOutput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.moriep import (
        MoriEPLLDispatchOutput,
        MoriEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


class AiterQuantType(str, Enum):
    NONE = "No"
    PER_TOKEN = "per_Token"
    PER_128X128 = "per_128x128"
    PER_1X32 = "per_1x32"


@dataclass
class AiterMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    quant_type: AiterQuantType = AiterQuantType.NONE
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    expert_mask: Optional[torch.Tensor] = None
    doweight_stage1: bool = False
    hidden_pad: int = 0
    intermediate_pad: int = 0
    swiglu_limit: float = 0.0
    fused_moe_kwargs: Optional[dict[str, Any]] = None


@dataclass
class AiterRunnerInput(RunnerInput):
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor  # int32
    topk_weights: torch.Tensor  # float32
    # Effective activation quant_type (may differ from quant_info.quant_type
    # after the dispatch-aware decision in mori pre_permute).
    quant_type: AiterQuantType
    # Per-token activation scale produced by an EP dispatcher (mori). Falls
    # back to quant_info.a13_scale when None.
    a1_scale: Optional[torch.Tensor] = None
    # Mori-only fused_moe kwargs.
    num_local_tokens: Optional[torch.Tensor] = None
    output_dtype: Optional[torch.dtype] = None
    output_tensor: Optional[torch.Tensor] = None

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


@dataclass
class AiterRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


_AITER_ACTIVATIONS = {
    "silu": "Silu",
    "swiglu": "Swiglu",
    "situ": "Situv2",
}


def _aiter_activation(activation: str):
    from aiter import ActivationType

    return getattr(ActivationType, _AITER_ACTIVATIONS.get(activation, "Gelu"))


def _aiter_quant_type(quant_type: AiterQuantType):
    from aiter import QuantType

    return getattr(QuantType, quant_type.value)


# CDNA2 has no pre-tuned AIter FP4 row, but its CK MXFP4 kernels can execute
# the DeepSeek-V4 TP4/EP4 shape when a concrete kernel is selected. Keep the
# names here instead of copying a gfx942 tune file: the kernels are generic
# CK FP4/BF16 kernels, while the assembly table is not valid on gfx90a.
_GFX90A_DSV4_FP4_KERNEL1 = (
    "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScale_v3_"
    "Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
)
_GFX90A_DSV4_FP4_KERNEL2 = (
    "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeight_v3_"
    "Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
)
_GFX90A_DSV4_FP4_KERNEL2_64THREAD = (
    "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeight_v1_"
    "Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
)


def _install_gfx90a_dsv4_fp4_tune(
    *,
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    topk: int,
    output_dtype: Optional[torch.dtype],
    quant_info: AiterMoeQuantInfo,
) -> None:
    """Register the validated CDNA2 DeepSeek-V4 FP4 CK fallback in AIter."""
    if quant_info.quant_type is not AiterQuantType.PER_1X32:
        return
    if hidden_states.device.type != "cuda" or w13_weight.device.type != "cuda":
        return
    if output_dtype not in (None, torch.bfloat16):
        return
    if quant_info.doweight_stage1:
        return

    try:
        import importlib
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx

        aiter_fused_moe = importlib.import_module("aiter.fused_moe")
    except (ImportError, RuntimeError):
        return

    if get_gfx() != "gfx90a":
        return

    # Avoid AIter's table lookup KeyError for non-tuned gfx90a shapes.
    if "gfx90a" not in aiter_fused_moe.fused_moe_1stage_dict:
        aiter_fused_moe.fused_moe_1stage_dict["gfx90a"] = (
            aiter_fused_moe.fused_moe_1stage_dict["gfx942"]
        )

    # Limit the local override to the two DSV4 rank layouts used on MI250X:
    # EP4/TP1 owns 64 full-width experts, while EP1/TP4 owns all 256 experts
    # with the intermediate dimension sharded four ways.  Both retain K=4096
    # and use the same generic CK tiles; only E and N differ.
    if w13_weight.dtype != getattr(torch, "float4_e2m1fn_x2", None):
        return
    if w13_weight.ndim != 3:
        return
    if tuple(w13_weight.shape) not in (
        (64, 4096, 2048),
        (256, 1024, 2048),
    ):
        return
    if topk != 6:
        return

    # gfx90a has no upstream per_1x32 entries. Avoid loading AIter's merged CSV
    # here: this function first runs during CUDA graph capture, and four ranks
    # serializing on AIter's temporary-file lock can enter Mori collectives at
    # different times and deadlock the capture. The exact validated shape above
    # is the only gfx90a entry this process needs.
    if aiter_fused_moe.cfg_2stages is None:
        aiter_fused_moe.cfg_2stages = {}

    padded_token = aiter_fused_moe.get_padded_M(int(hidden_states.shape[0]))
    model_dim = int(w13_weight.shape[2] * 2)
    inter_dim = int(w13_weight.shape[1] // 2)
    key = (
        int(get_cu_num()), padded_token, model_dim, inter_dim,
        int(w13_weight.shape[0]), int(topk), "ActivationType.Silu",
        str(output_dtype or torch.bfloat16), str(w13_weight.dtype),
        str(w13_weight.dtype), "QuantType.per_1x32", 1, 0,
    )
    if key not in aiter_fused_moe.cfg_2stages:
        from sglang.srt.environ import envs

        gfx90a_ksplit = envs.SGLANG_DSV4_GFX90A_AITER_MOE_KSPLIT.get()
        stage2_kernel = (
            _GFX90A_DSV4_FP4_KERNEL2_64THREAD
            if envs.SGLANG_DSV4_GFX90A_AITER_MOE_STAGE2_64THREAD.get()
            else _GFX90A_DSV4_FP4_KERNEL2
        )
        aiter_fused_moe.cfg_2stages[key] = {
            "block_m": 32,
            "ksplit": gfx90a_ksplit,
            # With split-K, leave the names empty so AIter selects its CKTile
            # BF16-activation/FP4-weight implementation. The validated legacy
            # CK pair remains the ksplit=0 baseline.
            "kernelName1": (
                "" if gfx90a_ksplit > 1 else _GFX90A_DSV4_FP4_KERNEL1
            ),
            "kernelName2": (
                "" if gfx90a_ksplit > 1 else stage2_kernel
            ),
            "run_1stage": 0,
        }
        aiter_fused_moe.get_2stage_cfgs.cache_clear()


@functools.cache
def _aiter_fused_moe_supports_no_combine() -> bool:
    """Probe whether the installed aiter.fused_moe accepts a `no_combine` kwarg.

    Older wheels don't expose it, so feature-detect once and forward
    conditionally, matching the existing `**extra` conditional-kwarg pattern
    used for `num_local_tokens` / `dtype`.
    """
    from aiter.fused_moe import fused_moe

    return "no_combine" in inspect.signature(fused_moe).parameters


@functools.cache
def _aiter_fused_moe_parameters() -> frozenset[str]:
    import inspect

    from aiter.fused_moe import fused_moe

    return frozenset(inspect.signature(fused_moe).parameters)


@functools.cache
def _install_gfx90a_aiter_quant_fallback() -> None:
    if is_gfx95_supported():
        return

    import importlib

    from aiter import QuantType
    from aiter.ops.quant import get_hip_quant, get_triton_quant

    fused_moe_module = importlib.import_module("aiter.fused_moe")
    triton_fp4_quant = get_triton_quant(QuantType.per_1x32)

    def get_quant(quant_type):
        if quant_type != QuantType.per_1x32:
            return get_hip_quant(quant_type)

        def quant_fp4(x, scale=None, quant_dtype=None, **kwargs):
            if x.dtype == quant_dtype and scale is not None:
                return x, scale
            return triton_fp4_quant(x, scale=None, quant_dtype=quant_dtype)

        return quant_fp4

    fused_moe_module.get_quant = get_quant


class AiterRunnerCore(MoeRunnerCore):
    def run(
        self,
        runner_input: AiterRunnerInput,
        quant_info: AiterMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> AiterRunnerOutput:
        if self.config.no_combine and not _aiter_fused_moe_supports_no_combine():
            raise NotImplementedError(
                "no_combine=True requested but the installed aiter.fused_moe does "
                "not accept a `no_combine` kwarg. Install an aiter build that "
                "supports fused_moe no_combine output."
            )

        if runner_input.hidden_states.shape[0] == 0:
            if self.config.no_combine:
                topk = runner_input.topk_ids.shape[-1]
                hidden_size = runner_input.hidden_states.shape[-1]
                return AiterRunnerOutput(
                    hidden_states=runner_input.hidden_states.new_empty(
                        (0, topk, hidden_size)
                    )
                )
            return AiterRunnerOutput(hidden_states=runner_input.hidden_states)

        from sglang.srt.environ import envs

        if (
            envs.SGLANG_DSV4_GFX90A_FP4_DIRECT_MOE.get()
            and runner_input.quant_type is AiterQuantType.PER_1X32
            and runner_input.hidden_states.dtype == torch.bfloat16
            and runner_input.hidden_states.ndim == 2
            and runner_input.hidden_states.shape[1] == 4096
            and runner_input.hidden_states.is_contiguous()
            and runner_input.topk_ids.ndim == 2
            and runner_input.topk_ids.shape[1] == 6
            and runner_input.topk_ids.dtype == torch.int32
            and runner_input.topk_ids.is_contiguous()
            and runner_input.topk_weights.shape == runner_input.topk_ids.shape
            and runner_input.topk_weights.dtype == torch.float32
            and runner_input.topk_weights.is_contiguous()
            and runner_input.num_local_tokens is not None
            and runner_input.num_local_tokens.shape == (1,)
            and runner_input.num_local_tokens.dtype == torch.int32
            and quant_info.w13_weight.shape == (64, 4096, 2048)
            and quant_info.w2_weight.shape == (64, 4096, 1024)
            and quant_info.w13_scale is not None
            and quant_info.w13_scale.numel() == 64 * 4096 * 128
            and quant_info.w13_scale.is_contiguous()
            and quant_info.w2_scale is not None
            and quant_info.w2_scale.numel() == 64 * 4096 * 64
            and quant_info.w2_scale.is_contiguous()
            and quant_info.expert_mask is not None
            and quant_info.expert_mask.shape == (256,)
            and quant_info.expert_mask.dtype == torch.int32
            and not getattr(quant_info.w13_weight, "is_shuffled", False)
            and not quant_info.doweight_stage1
            and quant_info.hidden_pad == 0
            and quant_info.intermediate_pad == 0
            and quant_info.b13 is None
            and quant_info.b2 is None
            and not self.config.no_combine
            and quant_info.swiglu_limit > 0
        ):
            from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
                gfx90a_fp4_expert_down,
                gfx90a_fp4_expert_gate_up,
            )

            intermediate = gfx90a_fp4_expert_gate_up(
                runner_input.hidden_states,
                quant_info.w13_weight,
                quant_info.w13_scale,
                runner_input.topk_ids,
                quant_info.expert_mask,
                runner_input.num_local_tokens,
                quant_info.swiglu_limit,
            )
            direct_out = (
                runner_input.output_tensor[
                    : runner_input.hidden_states.shape[0], :4096
                ]
                if runner_input.output_tensor is not None
                else None
            )
            output = gfx90a_fp4_expert_down(
                intermediate,
                quant_info.w2_weight,
                quant_info.w2_scale,
                runner_input.topk_ids,
                quant_info.expert_mask,
                runner_input.topk_weights,
                runner_input.num_local_tokens,
                out=direct_out,
            )
            return AiterRunnerOutput(hidden_states=output)
        elif envs.SGLANG_DSV4_GFX90A_FP4_DIRECT_MOE.get():
            logger.warning_once(
                "gfx90a direct FP4 MoE eligibility miss: "
                "hidden=%s/%s topk=%s/%s weights=%s,%s scales=%s,%s "
                "mask=%s/%s live=%s/%s pads=%s,%s shuffled=%s",
                tuple(runner_input.hidden_states.shape),
                runner_input.hidden_states.dtype,
                tuple(runner_input.topk_ids.shape),
                runner_input.topk_ids.dtype,
                tuple(quant_info.w13_weight.shape),
                tuple(quant_info.w2_weight.shape),
                None if quant_info.w13_scale is None else tuple(quant_info.w13_scale.shape),
                None if quant_info.w2_scale is None else tuple(quant_info.w2_scale.shape),
                None if quant_info.expert_mask is None else tuple(quant_info.expert_mask.shape),
                None if quant_info.expert_mask is None else quant_info.expert_mask.dtype,
                None if runner_input.num_local_tokens is None else tuple(runner_input.num_local_tokens.shape),
                None if runner_input.num_local_tokens is None else runner_input.num_local_tokens.dtype,
                quant_info.hidden_pad,
                quant_info.intermediate_pad,
                getattr(quant_info.w13_weight, "is_shuffled", False),
            )

        from aiter.fused_moe import fused_moe

        _install_gfx90a_aiter_quant_fallback()

        _install_gfx90a_dsv4_fp4_tune(
            hidden_states=runner_input.hidden_states,
            w13_weight=quant_info.w13_weight,
            topk=int(runner_input.topk_ids.shape[-1]),
            output_dtype=runner_input.output_dtype,
            quant_info=quant_info,
        )

        a1_scale = (
            runner_input.a1_scale
            if runner_input.a1_scale is not None
            else quant_info.a13_scale
        )

        extra: dict = {}
        if quant_info.fused_moe_kwargs:
            extra.update(quant_info.fused_moe_kwargs)
        if runner_input.num_local_tokens is not None:
            extra["num_local_tokens"] = runner_input.num_local_tokens
        if runner_input.output_dtype is not None:
            extra["dtype"] = runner_input.output_dtype
        fused_moe_parameters = _aiter_fused_moe_parameters()
        if "preshuffle" in fused_moe_parameters:
            extra["preshuffle"] = bool(
                getattr(quant_info.w13_weight, "is_shuffled", False)
            )
        if (
            runner_input.output_tensor is not None
            and "moe_out" in fused_moe_parameters
        ):
            extra["moe_out"] = runner_input.output_tensor
        if self.config.activation == "situ" and "gate_mode" in fused_moe_parameters:
            from aiter.ops.flydsl.moe_common import GateMode

            extra["gate_mode"] = GateMode.SEPARATED.value
            if self.config.gemm1_alpha is not None:
                extra["beta"] = float(self.config.gemm1_alpha)
            if self.config.gemm1_clamp_limit is not None:
                extra["linear_beta"] = float(self.config.gemm1_clamp_limit)
        elif quant_info.swiglu_limit > 0 and "gate_mode" in fused_moe_parameters:
            # GateMode is only needed for the gpt-oss MXFP4 swiglu_limit path.
            # Import lazily so models that don't use it (e.g. DeepSeek-V3 fp8,
            # swiglu_limit==0) still run on aiter builds where this module
            # lives elsewhere / is absent.
            from aiter.ops.flydsl.moe_common import GateMode

            # Default (INTERLEAVE) preserves the pre-fix behavior for paths
            # that prepare weights in the gate/up-interleaved layout. Set
            # `SGLANG_USE_AITER_MOE_GU_ITLV=0` to switch to SEPARATED, which
            # matches the layout produced by `Mxfp4MoEMethod` (gpt-oss
            # MXFP4) and the gptoss_fp4 tuned FlyDSL kernels.
            extra["gate_mode"] = (
                GateMode.INTERLEAVE.value
                if envs.SGLANG_USE_AITER_MOE_GU_ITLV.get()
                else GateMode.SEPARATED.value
            )
            extra["swiglu_limit"] = quant_info.swiglu_limit
        if self.config.no_combine:
            extra["no_combine"] = True

        try:
            output = fused_moe(
                hidden_states=runner_input.hidden_states,
                w1=quant_info.w13_weight,
                w2=quant_info.w2_weight,
                topk_weight=runner_input.topk_weights,
                topk_ids=runner_input.topk_ids,
                quant_type=_aiter_quant_type(runner_input.quant_type),
                activation=_aiter_activation(self.config.activation),
                w1_scale=quant_info.w13_scale,
                w2_scale=quant_info.w2_scale,
                a1_scale=a1_scale,
                a2_scale=quant_info.a2_scale,
                bias1=quant_info.b13,
                bias2=quant_info.b2,
                expert_mask=quant_info.expert_mask,
                doweight_stage1=quant_info.doweight_stage1,
                hidden_pad=quant_info.hidden_pad,
                intermediate_pad=quant_info.intermediate_pad,
                **extra,
            )
        except RuntimeError as exc:
            hs = runner_input.hidden_states
            local = runner_input.num_local_tokens
            local_value = (
                int(local.item())
                if local is not None and not torch.cuda.is_current_stream_capturing()
                else "captured"
            )
            raise RuntimeError(
                "AIter fused_moe rejected its dispatch contract: "
                f"hidden(shape={tuple(hs.shape)}, stride={hs.stride()}, "
                f"dtype={hs.dtype}, contiguous={hs.is_contiguous()}, "
                f"ptr_mod256={hs.data_ptr() % 256}); "
                f"topk_ids={tuple(runner_input.topk_ids.shape)}; "
                f"num_local_tokens={local_value}; "
                f"w1={tuple(quant_info.w13_weight.shape)} "
                f"w2={tuple(quant_info.w2_weight.shape)}; "
                f"quant_type={runner_input.quant_type}. Original error: {exc}"
            ) from exc
        return AiterRunnerOutput(hidden_states=output)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.AITER


# ---------------------------------------------------------------------------
# Pre-permute: dispatch_output -> AiterRunnerInput
# ---------------------------------------------------------------------------


@register_pre_permute("standard", "aiter")
def pre_permute_standard_to_aiter(
    dispatch_output: StandardDispatchOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AiterRunnerInput:
    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    topk_weights = topk_weights.to(torch.float32)

    if runner_config.apply_router_weight_on_input and not quant_info.doweight_stage1:
        # Pre-scale at the Python level for kernels that don't honor doweight_stage1.
        assert (
            topk_weights.dim() == 2 and topk_weights.shape[-1] == 1
        ), "apply_router_weight_on_input requires topk=1"
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        topk_weights = torch.ones_like(topk_weights)

    return AiterRunnerInput(
        hidden_states=hidden_states,
        topk_ids=topk_ids.to(torch.int32),
        topk_weights=topk_weights,
        quant_type=quant_info.quant_type,
    )


def _is_mori_dispatch_output(dispatch_output: Any) -> bool:
    # MoriEP{Normal,LL}DispatchOutput carry the post-mori-permute origin_topk_*
    # tensors that the standard DeepEP outputs lack.
    return hasattr(dispatch_output, "origin_topk_ids")


def _resolve_mori_quant_type(
    dispatch_a1_dtype: torch.dtype,
    dispatch_scale: Optional[torch.Tensor],
    weight_quant: AiterQuantType,
) -> AiterQuantType:
    """Pick the activation quant_type for AITER when the dispatch path may have
    pre-quantized hidden_states. Mirrors the original MoriEPMoE.run_moe_core
    decision tree."""
    is_fp8_quant = weight_quant in (
        AiterQuantType.PER_128X128,
        AiterQuantType.PER_TOKEN,
    )
    is_w4a4 = weight_quant == AiterQuantType.PER_1X32
    is_fp4_dispatch = dispatch_a1_dtype == torch.float4_e2m1fn_x2
    has_dispatch_scale = dispatch_scale is not None

    if is_w4a4:
        # W4A4 weights always run as per_1x32; FP8 dispatch is upscaled to BF16
        # before this point so dispatch_scale won't conflict.
        return AiterQuantType.PER_1X32
    if is_fp8_quant:
        return weight_quant
    # BF16 weights: lift to the dispatch-side quant type when scales are provided.
    if has_dispatch_scale and is_fp4_dispatch:
        return AiterQuantType.PER_1X32
    if has_dispatch_scale and not is_fp4_dispatch:
        return AiterQuantType.PER_128X128
    return AiterQuantType.NONE


def _pre_permute_deepep_to_aiter(
    dispatch_output: Union[
        DeepEPNormalDispatchOutput,
        DeepEPLLDispatchOutput,
        MoriEPNormalDispatchOutput,
        MoriEPLLDispatchOutput,
    ],
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> AiterRunnerInput:
    is_mori = _is_mori_dispatch_output(dispatch_output)

    hidden_states = dispatch_output.hidden_states
    topk_ids = dispatch_output.topk_ids.to(torch.int32)
    topk_weights = dispatch_output.topk_weights.to(torch.float32)
    a1_scale: Optional[torch.Tensor] = None
    num_local_tokens: Optional[torch.Tensor] = None
    output_dtype: Optional[torch.dtype] = None
    output_tensor: Optional[torch.Tensor] = None
    quant_type = quant_info.quant_type

    if is_mori:
        from sglang.kernels.ops.moe.rocm_moe_utils import upscale, upscale_mxfp4

        a1_scale = dispatch_output.hidden_states_scale
        num_local_tokens = dispatch_output.num_recv_tokens_per_expert
        output_dtype = dispatch_output.out_dtype
        output_tensor = dispatch_output.combine_output_buffer

        # Truncate dispatch tensors to the configured cap; mori combine only
        # reads [0, totalRecvTokenNum), so the truncated result needs no
        # padding back.
        mori_max = get_int_env_var("SGLANG_MORI_MOE_MAX_INPUT_TOKENS", 0)
        if mori_max > 0:
            hidden_states = hidden_states[:mori_max]
            if a1_scale is not None:
                a1_scale = a1_scale[:mori_max]
            topk_ids = topk_ids[:mori_max]
            topk_weights = topk_weights[:mori_max]

        # Upscale dispatched activations when there is no AITER kernel for the
        # weight/activation dtype pair.
        weight_quant = quant_info.quant_type
        is_fp8_quant = weight_quant in (
            AiterQuantType.PER_128X128,
            AiterQuantType.PER_TOKEN,
        )
        is_w4a4 = weight_quant == AiterQuantType.PER_1X32
        is_fp4_dispatch = hidden_states.dtype == torch.float4_e2m1fn_x2

        skip_gfx90a_prequant = get_bool_env_var(
            "SGLANG_GFX90A_AITER_MORI_SKIP_PREQUANT", "false"
        )

        if (
            is_w4a4
            and a1_scale is None
            and not is_fp4_dispatch
            and not is_gfx95_supported()
            and not skip_gfx90a_prequant
        ):
            # CDNA2 has no native FP4 conversion. AIter's HIP module_quant
            # emits unsupported FP8 instructions and its FP4 conversion is a
            # gfx950-only intrinsic. Quantize through AIter's Triton fallback
            # and pass pre-quantized activations to fused_moe.
            from aiter import QuantType
            from aiter.ops.quant import get_triton_quant

            hidden_states, a1_scale = get_triton_quant(QuantType.per_1x32)(
                hidden_states, shuffle=False
            )
            is_fp4_dispatch = True

        # AITER fused_moe Clamped-SwiGLU is dispatched with
        # gate_mode=INTERLEAVE, for which AITER picks a bf16/fp8 `q_dtype_a`
        # Refer to https://github.com/ROCm/aiter/blob/a2617c366dc7271a1662ecda2023d19f6ccefcec/aiter/fused_moe.py#L406-L412
        swiglu_interleave = quant_info.swiglu_limit > 0 and get_bool_env_var(
            "SGLANG_USE_AITER_MOE_GU_ITLV", "true"
        ) and "gate_mode" in _aiter_fused_moe_parameters()

        if is_w4a4 and a1_scale is not None and not is_fp4_dispatch:
            # W4A4 weights with FP8 dispatch: dequant FP8->BF16 first; the
            # FP4 per_1x32 path needs BF16 input.
            hidden_states = upscale(
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None
        elif is_w4a4 and is_fp4_dispatch and a1_scale is not None and swiglu_interleave:
            # W4A4 weights + FP4 dispatch on the clamped-SwiGLU/INTERLEAVE
            # path: AITER expects a bf16/fp8 activation here, not fp4x2.
            # Dequant FP4->BF16 and let fused_moe re-quantize internally.
            hidden_states = upscale_mxfp4(
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None
        elif is_fp8_quant and is_fp4_dispatch and a1_scale is not None:
            # FP8 weights + FP4 dispatch: no kernel for the fp4x2/fp8 pair;
            # dequant FP4->BF16 and let fused_moe re-quantize to FP8.
            hidden_states = upscale_mxfp4(
                hidden_states, a1_scale, num_local_tokens, output_dtype
            )
            a1_scale = None

        quant_type = _resolve_mori_quant_type(
            hidden_states.dtype, a1_scale, weight_quant
        )

        running_state["aiter_combine_topk_ids"] = dispatch_output.origin_topk_ids
        running_state["aiter_combine_topk_weights"] = (
            dispatch_output.origin_topk_weights
        )
    else:
        # DeepEP marks invalid topk slots with idx == -1; AITER cannot accept
        # negative ids, so reroute them to the sink slot at index
        # num_local_experts (masked off by quant_info.expert_mask which has
        # shape (num_local_experts + 1,)).
        topk_ids = torch.where(
            topk_ids == -1,
            torch.full_like(topk_ids, runner_config.num_local_experts),
            topk_ids,
        )
        running_state["aiter_combine_topk_ids"] = dispatch_output.topk_ids
        running_state["aiter_combine_topk_weights"] = dispatch_output.topk_weights

    running_state["aiter_combine_is_mori"] = is_mori

    return AiterRunnerInput(
        hidden_states=hidden_states,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        quant_type=quant_type,
        a1_scale=a1_scale,
        num_local_tokens=num_local_tokens,
        output_dtype=output_dtype,
        output_tensor=output_tensor,
    )


register_pre_permute("deepep_normal", "aiter")(_pre_permute_deepep_to_aiter)
register_pre_permute("deepep_ll", "aiter")(_pre_permute_deepep_to_aiter)


# ---------------------------------------------------------------------------
# Post-permute: AiterRunnerOutput -> CombineInput
# ---------------------------------------------------------------------------


@register_post_permute("aiter", "standard")
def post_permute_aiter_to_standard(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(hidden_states=runner_output.hidden_states)


def _post_permute_aiter_to_deepep(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
    is_normal: bool,
) -> CombineInput:
    if running_state.get("aiter_combine_is_mori"):
        from sglang.srt.layers.moe.token_dispatcher.moriep import (
            MoriEPLLCombineInput,
            MoriEPNormalCombineInput,
        )

        cls = MoriEPNormalCombineInput if is_normal else MoriEPLLCombineInput
    else:
        from sglang.srt.layers.moe.token_dispatcher.deepep import (
            DeepEPLLCombineInput,
            DeepEPNormalCombineInput,
        )

        cls = DeepEPNormalCombineInput if is_normal else DeepEPLLCombineInput

    return cls(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["aiter_combine_topk_ids"],
        topk_weights=running_state["aiter_combine_topk_weights"],
    )


@register_post_permute("aiter", "deepep_normal")
def post_permute_aiter_to_deepep_normal(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> CombineInput:
    return _post_permute_aiter_to_deepep(
        runner_output, quant_info, runner_config, running_state, is_normal=True
    )


@register_post_permute("aiter", "deepep_ll")
def post_permute_aiter_to_deepep_ll(
    runner_output: AiterRunnerOutput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> CombineInput:
    return _post_permute_aiter_to_deepep(
        runner_output, quant_info, runner_config, running_state, is_normal=False
    )
