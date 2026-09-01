"""Standalone-only TP4/M128 progressive all-reduce oracle."""

from __future__ import annotations

from torch.utils.cpp_extension import include_paths

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _jit_module():
    return load_jit(
        "gfx90a_tp4_m128_progressive_ar_oracle_v1",
        cuda_files=["deepseek_v4/gfx90a_tp4_m128_progressive_ar_oracle.cuh"],
        cuda_wrappers=[
            (
                "progressive",
                "sglang::Gfx90aTp4M128ProgressiveArOracle::progressive",
            ),
            (
                "wait_draft",
                "sglang::Gfx90aTp4M128ProgressiveArOracle::wait_draft",
            ),
            (
                "begin_draft",
                "sglang::Gfx90aTp4M128ProgressiveArOracle::begin_draft",
            ),
            (
                "anchor_end",
                "sglang::Gfx90aTp4M128ProgressiveArOracle::anchor_end",
            ),
            ("arm", "sglang::Gfx90aTp4M128ProgressiveArOracle::arm"),
        ],
        extra_cuda_cflags=["-O3", "-Rpass-analysis=kernel-resource-usage"],
        extra_include_paths=[
            *include_paths(),
            "/home/pc/pytorch/third_party/aiter/csrc/include",
            "/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/include",
        ],
    )


def progressive(fa, input, sync_workspace, output, rank):
    return _jit_module().progressive(fa, input, sync_workspace, output, rank)


def wait_draft(sync_workspace):
    return _jit_module().wait_draft(sync_workspace)


def begin_draft(fa, input, sync_workspace, output, rank):
    return _jit_module().begin_draft(fa, input, sync_workspace, output, rank)


def anchor_end(fa, input, routed, sync_workspace, output, rank):
    return _jit_module().anchor_end(
        fa, input, routed, sync_workspace, output, rank
    )


def arm(sync_workspace):
    return _jit_module().arm(sync_workspace)
