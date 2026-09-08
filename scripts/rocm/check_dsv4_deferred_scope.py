#!/usr/bin/env python3
"""CPU source-contract check for native M32 deferred-finalize integration."""
import ast
from pathlib import Path
import runpy

root = Path(__file__).resolve().parents[2]
# Reuse exhaustive negative topology/native/shape tests of the exact predicate.
runpy.run_path(str(root / 'scripts/rocm/check_dsv4_shared_after_topk_scope.py'))
model = ast.parse((root / 'python/sglang/srt/models/deepseek_v2.py').read_text())
method = next(n for n in ast.walk(model) if isinstance(n, ast.FunctionDef)
              and n.name == 'forward_normal_dual_stream')
source = ast.unparse(method)
assert source.index('run_gfx90a_deferred(') < source.index('mark(19)')
assert source.index('current_stream.wait_stream(self.alt_stream)') < source.index('final_hidden_states.finalize(shared_output)')
assert source.index('final_hidden_states.finalize(shared_output)') < source.index('tensor_model_parallel_all_reduce(final_hidden_states)')
gate = next(n for n in method.body if isinstance(n, ast.If)
            and 'TP8_M32_DEFERRED_FINALIZE' in ast.unparse(n.test))
condition = ast.unparse(gate.test)
for required in ('has_shared_output', 'not self._shared_expert_tp1', '_use_aiter',
                 'not use_flashinfer_trtllm_bypass', 'forward_batch is not None'):
    assert required in condition
assert 'shared_after_topk_eligible' in ast.unparse(gate)
runner = ast.parse((root / 'python/sglang/srt/layers/moe/moe_runner/aiter.py').read_text())
calls = [n for n in ast.walk(runner) if isinstance(n, ast.Call)
         and any(k.arg == 'defer_reduction' for k in n.keywords)]
assert len(calls) == 1 and ast.unparse(calls[0].func) == 'gfx90a_fp4_expert_down_grouped'
env = (root / 'python/sglang/srt/environ.py').read_text()
assert 'SGLANG_DSV4_GFX90A_TP8_M32_DEFERRED_FINALIZE = EnvBool(False)' in env
print('PASS: default-off, native topology guard, down-only flag, existing join/AR order')
