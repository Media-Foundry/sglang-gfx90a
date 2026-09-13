"""Explicitly enabled, bounded eager-only tensor capture for V4.1 diagnostics.

Use the existing --forward-hooks API. No module monkeypatches, no output
replacement, no sampling, and no extra collective is performed by this hook.
"""

import os
import hashlib
from pathlib import Path

import torch


def gather_packed_kv_rows(cache, ids, page_size):
    """Canonical [data576, scale8] records from page-planar FP8 KV storage.

    A page stores ALL 576-byte payloads before ALL 8-byte scales. A simple
    reshape(-1, 584) would silently capture the wrong bytes for most tokens.
    """
    raw = cache.view(torch.uint8).reshape(cache.shape[0], -1)
    ids = ids.long()
    assert bool(((ids >= 0) & (ids < raw.shape[0] * page_size)).all())
    page, offset = ids // page_size, ids % page_size
    data = raw[page[:, None], offset[:, None] * 576 + torch.arange(576, device=raw.device)]
    scales = raw[page[:, None], page_size * 576 + offset[:, None] * 8 + torch.arange(8, device=raw.device)]
    return torch.cat((data, scales), dim=-1)


def make_hook(config):
    folder = Path(config["folder"])
    label = config["label"]
    max_calls = int(config.get("max_calls", 2))
    capture_calls = config.get("capture_calls")
    if capture_calls is not None:
        capture_calls = set(capture_calls)
        if not capture_calls or any(type(i) is not int or not 0 <= i < max_calls for i in capture_calls):
            raise ValueError("capture_calls must select call indices within max_calls")
    max_tensor_bytes = int(config.get("max_tensor_bytes", 32 * 1024**2))
    summary_only = bool(config.get("summary_only", False))
    capture_parameters = bool(config.get("capture_parameters", True))
    min_rows = int(config.get("min_rows", 0))
    ranks = config.get("ranks")
    attention_contract_layers = set(config.get("attention_contract_layers", []))
    calls = {}
    module_numbers = {}

    def copy_tree(value, depth=0):
        if isinstance(value, torch.Tensor):
            if summary_only:
                cpu = value.detach().cpu().contiguous()
                digest = hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
                # Hash every byte, but retain only bounded rows for comparison.
                # This synchronizes the observed rank; never use these runs as
                # a timing or race-free-execution claim.
                if cpu.ndim and cpu.shape[0]:
                    indices = sorted({i for i in (0, 1, 63, 127, 511, 1023, cpu.shape[0]-2, cpu.shape[0]-1) if 0 <= i < cpu.shape[0]})
                    sample = cpu[indices].clone()
                else:
                    indices, sample = [], cpu.clone()
                return {"shape": list(cpu.shape), "dtype": str(cpu.dtype),
                        "sha256": digest, "sample_rows": indices, "sample": sample}
            if value.numel() * value.element_size() > max_tensor_bytes:
                return {"omitted_shape": list(value.shape), "dtype": str(value.dtype)}
            return value.detach().cpu().clone()
        if depth > 3:
            return {"type": type(value).__name__}
        if isinstance(value, (list, tuple)):
            return [copy_tree(v, depth + 1) for v in value]
        if isinstance(value, dict):
            return {str(k): copy_tree(v, depth + 1) for k, v in value.items()}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        # Runtime outputs use several non-dataclass structs. Capture only
        # named tensor contracts, never arbitrary object graphs or caches.
        fields = ("topk_ids", "topk_weights", "router_logits", "next_token_logits",
                  "input_ids", "positions", "req_pool_indices", "seq_lens",
                  "extend_seq_lens", "extend_prefix_lens")
        return {"type": type(value).__name__, **{
            name: copy_tree(getattr(value, name), depth + 1)
            for name in fields if hasattr(value, name)
        }}

    def hook(module, args, output):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if ranks is not None and rank not in ranks:
            return
        # Prefill-only collection can ignore scheduler run-ahead M1 decode.
        def first_tensor(value):
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, (tuple, list)):
                for child in value:
                    tensor = first_tensor(child)
                    if tensor is not None:
                        return tensor
            return None

        tensor = first_tensor(output)
        if tensor is None:
            tensor = first_tensor(args)
        if min_rows and (tensor is None or tensor.ndim == 0 or tensor.shape[0] < min_rows):
            return
        count = calls.get(id(module), 0)
        if count >= max_calls:
            return
        calls[id(module)] = count + 1
        if capture_calls is not None and count not in capture_calls:
            return
        module_number = module_numbers.setdefault(id(module), len(module_numbers))
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("V4.1 diagnostic hook must not run during graph capture")
        prefix = getattr(module, "prefix", "") or type(module).__name__
        target = folder / f"rank{rank}-{label}-{prefix}-module{module_number}-call{count}-pid{os.getpid()}.pt"
        folder.mkdir(parents=True, exist_ok=True)
        parameters = {
            name: copy_tree(param)
            for name, param in module.named_parameters(recurse=False)
        } if capture_parameters else {}
        record = {
            "rank": rank, "label": label, "prefix": prefix,
            "class": type(module).__name__, "call": count, "module_number": module_number,
            "layer_id": getattr(module, "layer_id", None), "summary_only": summary_only,
            "args": copy_tree(args), "output": copy_tree(output),
            "parameters": parameters,
        }
        if type(module).__name__ == "MQALayer" and getattr(module, "layer_id", None) in attention_contract_layers:
            # Snapshot the final valid query's *selected* packed cache rows,
            # not the whole allocated pool. Called before the next layer may
            # overwrite the metadata-owned Q pad. Diagnostic synchronization
            # only; never substitutes model outputs or changes KV state.
            from sglang.srt.model_executor.forward_context import get_attn_backend

            backend = get_attn_backend()
            meta = backend.forward_metadata
            core = meta.core_attn_metadata
            pool = backend.token_to_kv_pool
            q_pad = meta.q_pad_buffer
            swa = pool.get_swa_key_buffer_radix(module.layer_id)
            ids = core.swa_page_indices[-1].flatten().long()
            swa_count = int(core.swa_topk_lengths[-1])
            ids = ids[:swa_count]
            contract = {
                "q": q_pad[-1:].clone(), "q_batch_shape": list(q_pad.shape),
                "swa_ids": ids, "swa_kv": gather_packed_kv_rows(swa, ids, pool.swa_window_size),
                "swa_topk_capacity": core.swa_page_indices.shape[-1],
                "sink": module._local_attn_sink(), "softmax_scale": backend.softmax_scale,
                "position": core.positions_casual[-1:], "ratio": module.compress_ratio,
                "freqs_cis": module.freqs_cis[core.positions_casual[-1:].long()],
                "n_local_heads": module.n_local_heads, "n_local_groups": module.n_local_groups,
            }
            if count == 0:
                contract["wo_a_weight"] = module.wo_a.weight.view(module.n_local_groups, module.o_lora_rank, -1)
            from sglang.kernels.ops.attention.nsa_triton_decode import triton_mla_kernels_decode_fused as kernels
            contract["last_autotune_configs"] = {
                name: str(getattr(getattr(kernels, name), "best_config", None))
                for name in ("_fused_gather_attn_dsv4_kernel", "_fused_gather_attn_dsv4_dual_scope_kernel")
            }
            if module.compress_ratio in (1, 2):
                extra = pool.get_extra_key_buffer(module.layer_id)
                page_size = pool.page_size // module.compress_ratio
                ids = core.sparse_page_indices(module.compress_ratio)[-1].flatten().long()
                extra_count = int(core.sparse_topk_lengths(module.compress_ratio)[-1])
                ids = ids[:extra_count]
                valid = ids >= 0
                contract.update(extra_ids=ids, extra_valid=valid,
                                extra_topk_capacity=core.sparse_page_indices(module.compress_ratio).shape[-1],
                                extra_kv=gather_packed_kv_rows(extra, ids.clamp_min(0), page_size))
            record["attention_contract"] = copy_tree(contract)
        if summary_only and getattr(module, "compress_ratio", 0) in (1, 2):
            from sglang.srt.model_executor.forward_context import get_attn_backend

            core = get_attn_backend().forward_metadata.core_metadata
            ratio = module.compress_ratio
            record["sparse_metadata"] = copy_tree({
                "logical": core.sparse_raw_indices(ratio),
                "physical": core.sparse_page_indices(ratio),
                "positions": core.positions_casual,
            })
        with target.open("xb") as f:
            torch.save(record, f)
        return None

    return hook
