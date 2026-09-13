"""Explicitly enabled, bounded eager-only tensor capture for V4.1 diagnostics.

Use the existing --forward-hooks API. No module monkeypatches, no output
replacement, no sampling, and no extra collective is performed by this hook.
"""

import os
from pathlib import Path

import torch


def make_hook(config):
    folder = Path(config["folder"])
    label = config["label"]
    max_calls = int(config.get("max_calls", 2))
    max_tensor_bytes = int(config.get("max_tensor_bytes", 32 * 1024**2))
    calls = {}
    module_numbers = {}

    def copy_tree(value, depth=0):
        if isinstance(value, torch.Tensor):
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
        fields = ("topk_ids", "topk_weights", "router_logits", "next_token_logits")
        return {"type": type(value).__name__, **{
            name: copy_tree(getattr(value, name), depth + 1)
            for name in fields if hasattr(value, name)
        }}

    def hook(module, args, output):
        count = calls.get(id(module), 0)
        if count >= max_calls:
            return
        calls[id(module)] = count + 1
        module_number = module_numbers.setdefault(id(module), len(module_numbers))
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("V4.1 diagnostic hook must not run during graph capture")
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        prefix = getattr(module, "prefix", "") or type(module).__name__
        target = folder / f"rank{rank}-{label}-{prefix}-module{module_number}-call{count}-pid{os.getpid()}.pt"
        folder.mkdir(parents=True, exist_ok=True)
        parameters = {
            name: copy_tree(param)
            for name, param in module.named_parameters(recurse=False)
        }
        record = {
            "rank": rank, "label": label, "prefix": prefix,
            "class": type(module).__name__, "call": count, "module_number": module_number,
            "args": copy_tree(args), "output": copy_tree(output),
            "parameters": parameters,
        }
        with target.open("xb") as f:
            torch.save(record, f)
        return None

    return hook
