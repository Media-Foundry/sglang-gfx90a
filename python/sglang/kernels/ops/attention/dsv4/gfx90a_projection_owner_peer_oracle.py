from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.cpp_extension import include_paths

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "gfx90a_projection_owner_peer_oracle_v2",
        cuda_files=["deepseek_v4/gfx90a_projection_owner_peer_oracle.cuh"],
        cuda_wrappers=[
            ("publish", "sglang::Gfx90aProjectionOwnerPeerOracle::publish"),
            ("pack", "sglang::Gfx90aProjectionOwnerPeerOracle::pack"),
            ("end", "sglang::Gfx90aProjectionOwnerPeerOracle::end"),
        ],
        extra_cuda_cflags=["-O3"],
        extra_include_paths=[
            *include_paths(),
            "/home/pc/pytorch/third_party/aiter/csrc/include",
            "/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/include",
        ],
    )


def publish(
    fa: int,
    workspace: torch.Tensor,
    input: torch.Tensor,
    data: torch.Tensor,
    produced: torch.Tensor,
    rank: int,
) -> None:
    _jit_module().publish(fa, workspace, input, data, produced, rank)


def pack(
    fa: int,
    workspace: torch.Tensor,
    consumed: torch.Tensor,
    output: torch.Tensor,
    rank: int,
) -> None:
    _jit_module().pack(fa, workspace, consumed, output, rank)


def end(
    fa: int,
    workspace: torch.Tensor,
    end_epoch: torch.Tensor,
    rank: int,
) -> None:
    _jit_module().end(fa, workspace, end_epoch, rank)
