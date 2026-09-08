#!/usr/bin/env python3
"""CPU-only dispatch/cleanup tests; does not validate GPU arithmetic."""
import itertools
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization import dsv4_shared_gate_experiment as candidate


def main():
    keys = ("enabled", "hip", "arch", "native", "decode", "batch_size", "tp", "ep")
    choices = ((True, False), (True, False), ("gfx90a", "gfx942"),
               (True, False), (True, False), (1, 32), (8, 4), (1, 2))
    for values in itertools.product(*choices):
        assert candidate.eligible(**dict(zip(keys, values))) == all(
            value == alternatives[0] for value, alternatives in zip(values, choices)
        )
    flag = candidate.envs.SGLANG_DSV4_GFX90A_TP8_C1_SHARED_GATE_ROUND
    batch = SimpleNamespace(spec_algorithm=None, batch_size=1,
                            forward_mode=SimpleNamespace(is_decode=lambda: True))
    parallel = SimpleNamespace(tp_size=8, attn_tp_size=8, moe_ep_size=1)
    with patch.object(flag, "get", return_value=True), patch.object(
        torch.version, "hip", "mock"
    ), patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(
        gcnArchName="gfx90a:sramecc+:xnack-"
    )), patch("sglang.srt.runtime_context.get_parallel", return_value=parallel):
        with candidate.shared_gate_scope(batch, None):
            assert candidate._active.get()
            with patch.object(flag, "get", return_value=False):
                with candidate.shared_gate_scope(None, None):
                    assert not candidate._active.get()
            assert candidate._active.get()
            for field, value in (("batch_size", 32), ("spec_algorithm", SimpleNamespace(
                is_none=lambda: False
            )), ("forward_mode", SimpleNamespace(is_decode=lambda: False))):
                nested = SimpleNamespace(**{**vars(batch), field: value})
                with candidate.shared_gate_scope(nested, None):
                    assert not candidate._active.get()
                assert candidate._active.get()
            try:
                with candidate.shared_gate_scope(batch, None):
                    raise ValueError("test exception cleanup")
            except ValueError:
                pass
            assert candidate._active.get()
            x = torch.empty((1, 4096), dtype=torch.bfloat16)
            weight = torch.empty((512, 4096), dtype=torch.bfloat16)
            assert candidate.maybe_shared_gate(x, weight, 10) is None
        assert not candidate._active.get()
    assert candidate.maybe_shared_gate(None, None, None) is None
    # Execute the real forward's entry on a mock MLP. Its optional TP override
    # is None while the projection has resolved TP8, as in the actual service.
    source = Path(__file__).resolve().parents[2] / "python/sglang/srt/models/deepseek_v2.py"
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "DeepseekV2MLP")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    for arg in forward.args.args:
        arg.annotation = None
    forward.returns = None
    namespace = {"envs": candidate.envs}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(source), "exec"), namespace)
    sentinel = object()
    layer = SimpleNamespace(tp_size=None, swiglu_limit=10,
        gate_up_proj=SimpleNamespace(tp_size=8, _use_cached_block_fp8_bf16_weight=True,
                                    weight=object()), down_proj=lambda x: (x, None))
    with patch.object(flag, "get", return_value=True), patch.object(
        candidate, "maybe_shared_gate", return_value=sentinel
    ) as call:
        assert namespace["forward"](layer, object()) is sentinel
        assert call.call_count == 1
    print("PASS: 256 predicates; active/inactive nested scopes; speculative, prefill, "
          "C32 exclusions; exception cleanup; CPU rejection; resolved projection TP8")


if __name__ == "__main__":
    main()
