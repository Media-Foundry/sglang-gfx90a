"""Default-off, native TP8 M32 same-process down-kernel diagnostic."""

import logging

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.dsv4_ar_experiment import (
    down_uniform_capture,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
from sglang.srt.model_executor.runner_backend.paired_graph_transaction import capture_alternative

_pair = None
logger = logging.getLogger(__name__)


def switch_arm(enabled, *, idle):
    """Called by every TP worker for the same broadcast control request."""
    pair = _pair
    if pair is None:
        return False
    return pair.switch(enabled, idle=idle)


class DownGraphPair:
    EXTRA_LIMIT = 1024**3
    FREE_RESERVE = 2 * 1024**3

    def __init__(self, backend, *, memory_saver):
        global _pair
        from sglang.srt.configs.model_config import is_deepseek_v4
        from sglang.srt.distributed import get_moe_expert_parallel_world_size
        from sglang.srt.environ import envs

        runner = backend._cuda_graph_runner
        model = runner.model_runner
        args = model.server_args
        arch = (torch.cuda.get_device_properties(model.device).gcnArchName.split(':')[0]
                if torch.version.hip else '')
        valid = (
            type(runner).__name__ == 'DecodeCudaGraphRunner'
            and is_deepseek_v4(model.model_config.hf_config)
            and arch == 'gfx90a' and backend._tp_group.world_size == 8
            and get_moe_expert_parallel_world_size() == 1
            and model.spec_algorithm.is_none() and not model.is_draft_worker
            and args.max_total_tokens == 1048576
            and args.pp_size == 1 and args.dp_size == 1
            and not args.enable_dp_attention
            and not envs.SGLANG_DSV4_GFX90A_TP8_M32_DOWN_UNIFORM.get()
            and not memory_saver
            and not getattr(runner, 'enable_pdmux', False)
            and not runner.enable_torch_compile
            and not runner.ragged_verify_mode
            and not runner.enable_profile_cuda_graph
            and 1 in runner.capture_bs and 32 in runner.capture_bs
        )
        if not valid or _pair is not None:
            raise RuntimeError('paired down graphs require one native TP8/EP1 DSV4 runner, 1M KV, plain C1/M32 graphs')
        self.backend = backend
        self.key = None
        self.alternative = None
        self.enabled = False
        _pair = self

    def consensus(self, value):
        flag = torch.tensor([int(value)], dtype=torch.int32, device='cpu')
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.backend._tp_group.cpu_group)
        return bool(flag.item())

    def capture(self, key, capture, *, reset_after_capture):
        # Always capture the baseline arm, including C1 and all other tiers.
        with down_uniform_capture(False):
            capture()
        if key.size != 32:
            return
        # The existing baseline uses DSA dense/sparse dual graphs. Compare
        # only short-context dense M32; retain the sparse graph unchanged.
        if key.dsa_variant == 'sparse':
            return
        if (key.stream_idx is not None or key.variant_label is not None
                or key.dsa_variant not in (None, 'dense') or self.alternative is not None):
            raise RuntimeError(f'paired down capture accepts one plain/dense M32 key, got {key!r}')
        device = self.backend._device_module
        device.synchronize()
        free_before, _ = device.mem_get_info()
        if not self.consensus(free_before >= self.EXTRA_LIMIT + self.FREE_RESERVE):
            raise RuntimeError('paired graph free-memory admission failed; do not shrink KV')
        # Capture upgrades Raw -> Full on the host, but its metadata kernels
        # have only been recorded, not executed. A second eager warmup must
        # start from Raw again, just like the original warmup/capture boundary.
        if reset_after_capture is None:
            raise RuntimeError('paired DSV4 capture requires an attention metadata reset hook')
        reset_after_capture()
        pool = device.graph_pool_handle()
        with down_uniform_capture(True):
            alternative = capture_alternative(
                self.backend, key, pool=pool,
                set_allocator_pool=set_graph_pool_id, capture=capture,
            )
        device.synchronize()
        free_after, _ = device.mem_get_info()
        extra = max(0, free_before - free_after)
        if not self.consensus(extra <= self.EXTRA_LIMIT and free_after >= self.FREE_RESERVE):
            raise RuntimeError('paired graph exceeded 1 GiB/GCD diagnostic budget; do not shrink KV')
        self.key, self.alternative = key, alternative
        self.admission_free = free_before
        logger.info('DSV4 paired M32 captured: extra_device_bytes=%d free_bytes=%d kv_pool=1048576', extra, free_after)

    def selected(self, key):
        return self.alternative if self.enabled and key == self.key else None

    def switch(self, enabled, *, idle):
        if not self.consensus(type(enabled) is bool and idle and self.alternative is not None):
            return False
        self.backend._device_module.synchronize()
        # IPC registration happens after the outer capture scope closes.
        # Recheck before switching, conservatively including later tiers and
        # runtime allocations rather than counting only the private Torch pool.
        free_now, _ = self.backend._device_module.mem_get_info()
        if not self.consensus(free_now >= self.FREE_RESERVE
                              and self.admission_free - free_now <= self.EXTRA_LIMIT):
            logger.warning('DSV4 paired arm rejected by post-registration memory budget')
            return False
        self.backend._tp_group.barrier()
        self.enabled = enabled
        logger.info('DSV4 paired M32 arm=%s; baseline graph mappings retained', enabled)
        return True

    def cleanup(self):
        global _pair
        self.alternative = None
        self.enabled = False
        if _pair is self:
            _pair = None
