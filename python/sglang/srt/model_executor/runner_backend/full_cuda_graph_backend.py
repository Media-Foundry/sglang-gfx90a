# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FullCudaGraphBackend — captures the entire model forward as one
torch.cuda.CUDAGraph per shape.
"""

from __future__ import annotations

import ctypes
from contextlib import AbstractContextManager, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_memory_pool,
    graph_pool_capture_scope,
    graph_pool_replay_scope,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


class FullCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.cuda.CUDAGraph per shape; attention metadata is
    captured inside the graph. Memory-saver-aware.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
    ) -> None:
        self._graphs: Dict[Any, torch.cuda.CUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._cuda_graph_runner = cuda_graph_runner
        self._device_module = cuda_graph_runner.device_module
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )

    @staticmethod
    def _maybe_upload_rocm_graph(graph: torch.cuda.CUDAGraph) -> None:
        """Pre-upload a HIP graph exec once, outside latency-critical replay."""
        if not torch.version.hip or not get_bool_env_var(
            "SGLANG_ROCM_CUDA_GRAPH_UPLOAD"
        ):
            return

        # ROCm 7.14 documents hipGraphInstantiateWithFlags as ignoring all
        # flags, including hipGraphInstantiateFlagUpload. PyTorch therefore
        # cannot request upload while it instantiates the graph; invoke the
        # explicit runtime API after capture instead.
        hip = ctypes.CDLL("libamdhip64.so")
        upload = hip.hipGraphUpload
        upload.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        upload.restype = ctypes.c_int
        stream = torch.cuda.current_stream()
        status = upload(
            ctypes.c_void_p(graph.raw_cuda_graph_exec()),
            ctypes.c_void_p(stream.cuda_stream),
        )
        if status != 0:
            raise RuntimeError(f"hipGraphUpload failed with HIP status {status}")
        stream.synchronize()

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            self._pool = get_or_create_global_graph_memory_pool(self._device_module)
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # When per-bs capture traces are enabled (--enable-profile-cuda-graph +
        # SGLANG_GRAPH_BATCH_CAPTURE), the runner created a scheduled
        # torch profiler (wait=2, active=1) and exposed it as _profiler. We step()
        # past the two warmup runs so only the capture run is recorded, and each
        # batch size produces its own trace via the profiler's on_trace_ready.
        # With --enable-profile-cuda-graph alone the runner leaves _profiler None
        # (its unscheduled profiler records the whole capture in one pass), so no
        # stepping happens here.
        runner = self._cuda_graph_runner
        profiler = (
            getattr(runner, "_profiler", None)
            if getattr(runner, "enable_profile_cuda_graph", False)
            else None
        )

        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if profiler is not None:
                profiler.step()
            if post_warmup_hook is not None:
                post_warmup_hook()

        # forward_fn is asynchronous. Without a final drain, faster ranks can
        # enter HIP graph capture while peers are still inside the last Mori
        # warmup collective. A faster gfx90a MHC pre-mix made this latent race
        # deterministic. Keep capture start rank-aligned; this is startup-only
        # and adds no replay/decode overhead.
        self._device_module.synchronize()
        self._tp_group.barrier()

        graph = torch.cuda.CUDAGraph()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = self._device_module.graph

        with (
            graph_pool_capture_scope(),
            graph_ctx(cuda_graph=graph, pool=self._pool, stream=self._capture_stream),
        ):
            out = forward_fn()

        if profiler is not None:
            profiler.step()

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out
        self._maybe_upload_rocm_graph(graph)

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        with graph_pool_replay_scope():
            self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
