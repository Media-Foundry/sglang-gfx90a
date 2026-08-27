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
        "gfx90a_tile_epoch_pipeline_oracle_v7",
        cuda_files=["deepseek_v4/gfx90a_tile_epoch_pipeline_oracle.cuh"],
        cuda_wrappers=[
            ("producer", "sglang::Gfx90aTileEpochPipelineOracle::producer"),
            ("reduce", "sglang::Gfx90aTileEpochPipelineOracle::reduce"),
            ("end", "sglang::Gfx90aTileEpochPipelineOracle::end"),
            ("wait_only", "sglang::Gfx90aTileEpochPipelineOracle::wait_only"),
            ("load_only", "sglang::Gfx90aTileEpochPipelineOracle::load_only"),
            ("ack", "sglang::Gfx90aTileEpochPipelineOracle::ack"),
        ],
        extra_cuda_cflags=["-O3"],
        extra_include_paths=[
            *include_paths(),
            "/home/pc/pytorch/third_party/aiter/csrc/include",
            "/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/include",
        ],
    )


def producer(fa: int, workspace: torch.Tensor, data: torch.Tensor, produced: torch.Tensor,
             consumed: torch.Tensor, rank: int) -> None:
    _jit_module().producer(fa, workspace, data, produced, consumed, rank)


def reduce(fa: int, workspace: torch.Tensor, data: torch.Tensor, produced: torch.Tensor,
           consumed: torch.Tensor, output: torch.Tensor, rank: int) -> None:
    _jit_module().reduce(fa, workspace, data, produced, consumed, output, rank)


def end(fa: int, workspace: torch.Tensor, end_epoch: torch.Tensor, rank: int) -> None:
    _jit_module().end(fa, workspace, end_epoch, rank)


def wait_only(fa: int, workspace: torch.Tensor, produced: torch.Tensor, consumed: torch.Tensor,
              waited: torch.Tensor, rank: int) -> None:
    _jit_module().wait_only(fa, workspace, produced, consumed, waited, rank)


def load_only(fa: int, workspace: torch.Tensor, data: torch.Tensor, output: torch.Tensor,
              epoch: int) -> None:
    _jit_module().load_only(fa, workspace, data, output, epoch)


def ack(consumed: torch.Tensor, rank: int) -> None:
    _jit_module().ack(consumed, rank)
