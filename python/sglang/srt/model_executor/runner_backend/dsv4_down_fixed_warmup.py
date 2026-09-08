"""Startup-only diagnostic: equalize down-module first use without extra graphs."""

import logging

logger = logging.getLogger(__name__)


def memory_consensus(group, valid):
    import torch
    import torch.distributed as dist

    flag = torch.tensor([int(valid)], dtype=torch.int32, device='cpu')
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group.cpu_group)
    return bool(flag.item())


def validate_runner(backend, *, memory_saver):
    import torch

    from sglang.srt.configs.model_config import is_deepseek_v4
    from sglang.srt.distributed import get_moe_expert_parallel_world_size

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
        and not args.enable_dp_attention and not memory_saver
        and not getattr(runner, 'enable_pdmux', False)
        and not runner.enable_torch_compile and not runner.ragged_verify_mode
        and not runner.enable_profile_cuda_graph
        and 1 in runner.capture_bs and 32 in runner.capture_bs
        and backend._down_graph_pair is None
    )
    if not valid:
        raise RuntimeError('fixed down warmup requires native TP8/EP1 DSV4, 1M KV, single C1/M32 graphs')


def warmup_pair(key, forward, reset, *, device, group, override):
    """No returned tensors, graph capture, pool changes, or persistent buffers.

    Each arm is drained before restoring Raw attention metadata. A failed
    forward aborts startup; it must not be retried in a possibly poisoned HIP
    context. The context manager still restores the selector on failure.
    """
    if key.size != 32 or key.dsa_variant == 'sparse':
        return False
    if (key.stream_idx is not None or key.variant_label is not None
            or key.dsa_variant not in (None, 'dense') or reset is None):
        raise RuntimeError('fixed down warmup requires a plain/dense M32 key and metadata reset')
    device.synchronize()
    group.barrier()
    free_before, _ = device.mem_get_info()
    reserve, limit = 2 * 1024**3, 256 * 1024**2
    if not memory_consensus(group, free_before >= reserve + limit):
        raise RuntimeError('fixed down warmup memory admission failed; do not shrink KV')
    # The regular selected-arm warmups and capture still run after this pair.
    for arm in (False, True):
        with override(arm):
            reset()
            group.barrier()
            forward()
            device.synchronize()
            reset()
            if arm is False:
                # First normal forward allocates ordinary model workspaces.
                # Charge only the subsequent alternate arm to this experiment.
                free_baseline, _ = device.mem_get_info()
    group.barrier()
    free_after, _ = device.mem_get_info()
    if not memory_consensus(group, free_after >= reserve and free_baseline - free_after <= limit):
        raise RuntimeError('fixed down warmup exceeded diagnostic memory budget; do not shrink KV')
    logger.info('DSV4 fixed down first-use order=baseline,candidate baseline_warmup_bytes=%d extra_device_bytes=%d free_bytes=%d',
                free_before - free_baseline, free_baseline - free_after, free_after)
    return True
