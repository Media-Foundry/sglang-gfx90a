"""Explicitly enabled, bounded eager-only tensor capture for V4.1 diagnostics.

Use the existing --forward-hooks API. No module monkeypatches, no output
replacement, no sampling, and no extra collective is performed by this hook.
"""

import os
import hashlib
from pathlib import Path

import torch


def make_hook(config):
    folder = Path(config["folder"])
    label = config["label"]
    max_calls = int(config.get("max_calls", 2))
    max_tensor_bytes = int(config.get("max_tensor_bytes", 32 * 1024**2))
    summary_only = bool(config.get("summary_only", False))
    capture_parameters = bool(config.get("capture_parameters", True))
    min_rows = int(config.get("min_rows", 0))
    ranks = config.get("ranks")
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
