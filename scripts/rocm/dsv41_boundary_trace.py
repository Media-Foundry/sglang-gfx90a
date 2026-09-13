"""Opt-in eager diagnostic wrappers; original functions run exactly once.

Installed only through an explicit forward-hook factory. CPU snapshots add
synchronization: these traces cannot establish timings or absence of races.
No returned tensor, weight, cache entry or collective is replaced.
"""

import functools
import inspect
import os
from pathlib import Path

import torch


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, (tuple, list)):
        return [snapshot(x) for x in value]
    if isinstance(value, dict):
        return {k: snapshot(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__}


def wrap_method(original, tag, save, selected, max_calls):
    signature = inspect.signature(original)
    calls = {}

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        count = calls.get(id(self), 0)
        if not selected(self) or count >= max_calls:
            return original(self, *args, **kwargs)
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Boundary diagnostics require eager execution")
        calls[id(self)] = count + 1
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        inputs = {k: snapshot(v) for k, v in bound.arguments.items() if k != "self"}
        if tag == "engram_gate":
            inputs.update(q_weight=snapshot(self.q_weight), k_weight=snapshot(self.k_weight),
                          eps=self.eps, clamp_value=self.clamp_value)
        output = original(self, *args, **kwargs)
        save(self, tag, count, inputs, snapshot(output))
        return output

    return wrapped


def make_hook(config):
    from sglang.srt.models import deepseek_v41 as model
    from sglang.srt.layers.engram import Engram

    if getattr(model, "_boundary_trace_installed", False):
        raise RuntimeError("Only one V4.1 boundary trace factory may be installed")
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    folder = Path(config["folder"])
    layers = set(config.get("layers", [12, 13, 20, 21, 22]))
    ranks = set(config.get("ranks", [0]))
    max_calls = int(config.get("max_calls", 10))
    if max_calls < 1 or not layers or not ranks:
        raise ValueError("Trace selection and call bound must be nonempty")
    numbers = {}

    def save(module, tag, count, inputs, output):
        number = numbers.setdefault(id(module), len(numbers))
        layer = getattr(module, "layer_id", None)
        prefix = getattr(getattr(module, "wkv", None), "prefix", "")
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"rank{rank}-{tag}-layer{layer}-module{number}-call{count}.pt"
        with target.open("xb") as f:
            torch.save({"rank": rank, "tag": tag, "layer_id": layer,
                        "prefix": prefix, "call": count,
                        "args": inputs, "output": output}, f)

    selected = lambda self: rank in ranks and self.layer_id in layers
    cls = model.DeepseekV4DecoderLayer
    for name in ("_hc_mix_and_combine", "hc_post"):
        setattr(cls, name, wrap_method(getattr(cls, name), name, save, selected, max_calls))
    # Engram does not store the model layer_id; retain its projection prefix
    # and a per-instance number, and capture each Engram rather than guessing.
    Engram.apply_gate = wrap_method(Engram.apply_gate, "engram_gate", save,
                                   lambda self: rank in ranks, max_calls // 2 or 1)
    model._boundary_trace_installed = True
    return lambda module, args, output: None
