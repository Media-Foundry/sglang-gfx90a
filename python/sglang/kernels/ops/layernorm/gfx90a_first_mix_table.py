"""Opt-in original-V4 first-layer FP32 mixes, derived from local vocab shards.

Only plain token embeddings in native TP8 eager prefill are admitted. This
does not cache hidden activations or later-layer state. Online updater APIs
reject this experimental derived cache; model reload hooks invalidate it.
"""
from dataclasses import dataclass
import logging
import os

import torch

from sglang.kernels.ops.layernorm.gfx90a_mhc_prefill_policy import common_unfused_fp32

ENV = "SGLANG_DSV4_PREFILL_FIRST_MIX_TABLE"
CHECK_ENV = "SGLANG_DSV4_DEBUG_FIRST_MIX_TABLE_CHECK"
ENTRY = "_dsv4_first_mhc_mixes"
CACHE = "_dsv4_first_mix_cache"
logger = logging.getLogger(__name__)


def eligible(model, batch, input_embeds):
    if os.getenv(ENV, "0") != "1" or input_embeds is not None:
        return False
    if getattr(batch.forward_mode, "name", None) != "EXTEND":
        return False
    if any(getattr(batch, "lora_ids", None) or []):
        return False
    if any(item is not None for item in (getattr(batch, "mm_inputs", None) or [])):
        return False
    if model.start_layer != 0 or model.end_layer != 43:
        return False
    cfg = model.layers[0].config
    if getattr(cfg, "model_type", None) != "deepseek_v4":
        return False
    return common_unfused_fp32(batch.input_ids.shape[0])


def shard_contract(embedding, rank, world):
    """Plain contiguous base-vocabulary shards only; no adapter vocabulary."""
    assert world == embedding.tp_size == 8
    assert embedding.num_added_embeddings == 0
    assert embedding.num_embeddings == embedding.org_vocab_size == 129280
    padded = embedding.num_embeddings_padded
    assert padded == embedding.org_vocab_size_padded and padded % world == 0
    local = padded // world
    assert embedding.num_embeddings_per_partition == local
    shard = embedding.shard_indices
    assert shard.padded_org_vocab_start_index == rank * local
    assert shard.padded_org_vocab_end_index == (rank + 1) * local
    assert shard.org_vocab_start_index == min(rank * local, 129280)
    assert shard.org_vocab_end_index == min((rank + 1) * local, 129280)
    return local, padded


def weight_key(embedding, fn, epsilon):
    def one(value):
        return (id(value), value.data_ptr(), value._version, tuple(value.shape), value.dtype, value.device)
    return one(embedding.weight), one(fn), epsilon, embedding.num_embeddings_padded


@dataclass
class Cache:
    key: tuple
    table: torch.Tensor
    weight: torch.Tensor  # Keep identities alive; no copy of embedding weights.
    fn: torch.Tensor
    ready: torch.cuda.Event


def invalidate(model):
    setattr(model, CACHE, None)


def build(embedding, fn, epsilon, group):
    from sglang.srt.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
    from sglang.kernels.ops.layernorm.gfx90a_mhc_pre_mix import _jit_gfx90a_mhc_pre_mix_module

    local_rows, padded = shard_contract(embedding, group.rank_in_group, group.world_size)
    weight = embedding.weight
    assert type(embedding.quant_method) is UnquantizedEmbeddingMethod
    assert weight.shape == (local_rows, 4096) and weight.dtype == torch.bfloat16
    assert fn.shape == (24, 16384) and fn.dtype == torch.float32
    assert weight.device == fn.device and weight.is_cuda
    assert weight.is_contiguous() and fn.is_contiguous()
    assert embedding.output_dtype in (None, torch.bfloat16)
    assert torch.version.hip and "gfx90a" in torch.cuda.get_device_properties(weight.device).gcnArchName
    assert not torch.cuda.is_current_stream_capturing()
    local = torch.empty((local_rows, 1, 24), device=weight.device, dtype=torch.float32)
    module = _jit_gfx90a_mhc_pre_mix_module()
    for lo in range(0, local_rows, 1024):
        hi = min(lo + 1024, local_rows)
        residual = weight[lo:hi].unsqueeze(1).repeat(1, 4, 1)
        module.run(residual, fn, local[lo:hi], epsilon)
    table = torch.empty((padded, 1, 24), device=weight.device, dtype=torch.float32)
    # One startup copy collective; deliberately no custom IPC allocator assumptions.
    torch.distributed.all_gather_into_tensor(table, local, group=group.device_group)
    ready = torch.cuda.Event()
    ready.record()
    logger.info("DSV4 first-mix table built: rank=%d local_vocab=%d padded_vocab=%d bytes=%d",
                group.rank_in_group, local_rows, padded, table.numel() * table.element_size())
    return Cache(weight_key(embedding, fn, epsilon), table, weight, fn, ready)


def prepare(model, input_ids, batch, input_embeds, residual):
    batch.__dict__.pop(ENTRY, None)
    if not eligible(model, batch, input_embeds):
        return
    from sglang.srt.distributed import get_tp_group
    from sglang.srt.environ import envs

    assert os.getenv("SGLANG_DSV4_GFX90A_TRITON_MHC_PRE_MIX", "0") == "1"
    assert envs.SGLANG_DSV4_GFX90A_WAVE64_MHC_PRE_MIX.get()
    assert not envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get()
    assert not envs.SGLANG_OPT_USE_FLASHINFER_MHC.get()
    assert not envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get()
    assert not os.getenv("SGLANG_DSV4_DEBUG_FORCE_TORCH_MHC")
    assert not torch.cuda.is_current_stream_capturing()
    embedding = model.embed_tokens
    layer = model.layers[0]
    fn = layer.hc_attn_fn
    epsilon = layer.rms_norm_eps
    key = weight_key(embedding, fn, epsilon)
    cache = getattr(model, CACHE, None)
    if cache is None:
        cache = build(embedding, fn, epsilon, get_tp_group())
        setattr(model, CACHE, cache)
    elif cache.key != key:
        # Never start a collective rebuild on only the rank whose shard changed.
        raise RuntimeError("First-MHC weights changed without rank-uniform cache invalidation; reload/restart")
    stream = torch.cuda.current_stream(input_ids.device)
    stream.wait_event(cache.ready)
    cache.table.record_stream(stream)
    assert input_ids.ndim == 1 and input_ids.device == cache.table.device
    assert residual.shape == (input_ids.numel(), 4, 4096)
    torch._assert_async(((input_ids >= 0) & (input_ids < embedding.num_embeddings)).all(),
                        "Invalid token ID for first-MHC table")
    mixes = cache.table[input_ids]
    checking = os.getenv(CHECK_ENV, "0") == "1"
    if checking:
        from sglang.kernels.ops.layernorm.gfx90a_mhc_pre_mix import _jit_gfx90a_mhc_pre_mix_module
        reference = torch.empty_like(mixes)
        module = _jit_gfx90a_mhc_pre_mix_module()
        (module.run_m64 if len(input_ids) == 64 else module.run)(residual, fn, reference, epsilon)
        assert torch.equal(mixes.view(torch.uint8), reference.view(torch.uint8)), "First-mix table mismatch"
        logger.info("DSV4 first-mix table exact: rank=%d rows=%d", get_tp_group().rank_in_group, len(input_ids))
    batch.__dict__[ENTRY] = (fn, mixes)


def take(layer, batch, fn, rows):
    if layer != 0 or batch is None:
        return None
    entry = batch.__dict__.pop(ENTRY, None)
    if entry is None:
        return None
    owner, mixes = entry
    assert owner is fn and mixes.shape == (rows, 1, 24)
    if os.getenv(CHECK_ENV, "0") == "1":
        from sglang.srt.distributed import get_tp_group
        logger.info("DSV4 first-mix table consumed: rank=%d rows=%d", get_tp_group().rank_in_group, rows)
    return mixes
