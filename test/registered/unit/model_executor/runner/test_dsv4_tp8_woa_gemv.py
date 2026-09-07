"""CPU-only shape/dispatch guards for TP8's one-group wo_a specialization."""
import ast
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from sglang.kernels.ops.quantization import gfx90a_bf16_gemv as mod


def tensor(shape):
    return SimpleNamespace(shape=shape, ndim=len(shape), dtype=torch.bfloat16,
                           device="cuda", is_contiguous=lambda: True)


class TestTP8WoaGemv(unittest.TestCase):
    def test_shape_guard_and_existing_tp4(self):
        for m, g, allow, selected in [
            (1,1,False,False), (1,1,True,True), (2,1,True,False),
            (1,2,False,True), (8,2,False,True), (9,2,True,False),
            (1,4,True,False),
        ]:
            x,w = tensor((m,g,4096)),tensor((g,1024,4096))
            with patch.object(torch.version,"hip","test"), patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(gcnArchName="gfx90a"),
            ), patch.object(mod,"_jit_gfx90a_bf16_grouped_gemv_module") as jit, patch(
                "torch.empty",return_value=object()
            ) as alloc:
                y=mod.gfx90a_wave64_bf16_grouped_gemv(x,w,allow_single_group=allow)
                if selected:
                    self.assertIs(y,alloc.return_value)
                    jit.assert_called_once_with(m,g)
                    jit.return_value.run.assert_called_once_with(x,w,y)
                else:
                    self.assertIsNone(y)
                    jit.assert_not_called()
                    alloc.assert_not_called()

    def test_model_gate_is_native_decode_only(self):
        path=Path(__file__).resolve().parents[5]/"python/sglang/srt/models/deepseek_v4.py"
        tree=ast.parse(path.read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Name)
               and n.func.id=="gfx90a_wave64_bf16_grouped_gemv"]
        self.assertEqual(len(calls),1)
        expr=next(k.value for k in calls[0].keywords if k.arg=="allow_single_group")
        code=compile(ast.Expression(expr),str(path),"eval")
        for flag,tp,bs,mode in itertools.product(
            (False,True),(4,8),(1,2,32),("decode","extend","target_verify","draft_extend","idle")
        ):
            actual=eval(code,{
                "envs":SimpleNamespace(SGLANG_DSV4_GFX90A_TP8_BS1_WOA_GEMV=SimpleNamespace(get=lambda:flag)),
                "self":SimpleNamespace(attn_tp_size=tp),
                "forward_batch":SimpleNamespace(batch_size=bs,forward_mode=SimpleNamespace(is_decode=lambda:mode=="decode")),
            })
            self.assertEqual(actual,flag and tp==8 and bs==1 and mode=="decode")


if __name__=="__main__":
    unittest.main()
