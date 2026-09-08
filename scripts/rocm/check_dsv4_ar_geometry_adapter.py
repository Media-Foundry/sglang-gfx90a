#!/usr/bin/env python3
"""CPU contract tests for actual adapter source; GPU oracle is separate."""
import ast
from contextvars import ContextVar
import logging
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / 'python/sglang/srt/distributed/device_communicators/dsv4_ar_geometry.py'


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.blocks, self.capture, self.calls = 4, False, []
        self.scope = ContextVar('test_scope', default=False)
        self.args = NS(tp_size=8, ep_size=1, dp_size=1, pp_size=1,
                       enable_dp_attention=False, speculative_algorithm=None,
                       max_total_tokens=1048576)
        owner = self
        class Parent:
            def __init__(self):
                self.disabled, self.world_size, self.device, self._ptr = False, 8, 0, 123
            def all_reduce(self, inp, **kwargs):
                owner.calls.append(('parent', inp, kwargs))
                return 'parent'
        self.parent = Parent
        self.tensor = lambda shape=(32,4096): NS(shape=shape, dtype='bf16', device=0, is_contiguous=lambda: True)
        torch = NS(bfloat16='bf16', empty_like=lambda x: self.tensor(x.shape), version=NS(hip=True),
                   cuda=NS(is_current_stream_capturing=lambda: self.capture,
                           get_device_properties=lambda _: NS(gcnArchName='gfx90a')))
        self.module = NS(signal_bytes=lambda: 16,
                         run=lambda *a: self.calls.append(('kernel', a)))
        self.modules = {
            'sglang.srt.server_args': NS(get_global_server_args=lambda: self.args),
            'aiter': NS(meta_size=lambda: 16),
            'sglang.kernels.ops.debug.gfx90a_tp8_ar_geometry_oracle': NS(module=lambda: self.module),
        }
        tree = ast.parse(SOURCE.read_text())
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        self.ns = dict(torch=torch, logging=logging, _active=self.scope,
                       envs=NS(SGLANG_DSV4_GFX90A_TP8_M32_AR_BLOCKS=NS(get=lambda: self.blocks),
                               SGLANG_DSV4_GFX90A_TP8_M32_LEGACY_AR=NS(get=lambda: True)))
        exec(compile(tree, str(SOURCE), 'exec'), self.ns)

    def instance(self):
        with patch.dict('sys.modules', self.modules):
            return self.ns['adapt_geometry'](self.parent)()

    def test_off_is_identity(self):
        self.blocks = 0
        self.assertIs(self.ns['adapt_geometry'](self.parent), self.parent)

    def test_capture_preserves_output_and_uses_existing_handle(self):
        ar = self.instance()
        self.capture = True
        self.scope.set(True)
        x, out = self.tensor(), self.tensor()
        self.assertIs(ar.all_reduce(x, out=out, registered=True), out)
        self.assertEqual(self.calls, [('kernel', (123, x, out, 4))])

    def test_all_fallback_modes_forward_unchanged(self):
        ar = self.instance()
        for active, capture, registered, shape, quant in (
            (False, True, True, (32,4096), False),
            (True, False, True, (32,4096), False),
            (True, True, False, (32,4096), False),
            (True, True, True, (1,4096), False),
            (True, True, True, (32,4096), True),
        ):
            self.scope.set(active)
            self.capture = capture
            x, out = self.tensor(shape), self.tensor(shape)
            self.assertEqual(ar.all_reduce(x,out=out,use_new=False,registered=registered,
                                          open_fp8_quant=quant), 'parent')
            self.assertEqual(self.calls[-1], ('parent', x, dict(out=out,use_new=False,
                                   registered=registered,open_fp8_quant=quant)))

    def test_bad_scope_rejected_before_loading(self):
        for key, value in [('tp_size',4),('ep_size',2),('speculative_algorithm','EAGLE'),
                           ('max_total_tokens',131072),('enable_dp_attention',True)]:
            old = getattr(self.args, key)
            setattr(self.args,key,value)
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.instance()
            setattr(self.args,key,old)

    def test_abi_failure_is_fatal(self):
        self.module.signal_bytes = lambda: 32
        with self.assertRaisesRegex(RuntimeError, 'ABI'):
            self.instance()


if __name__ == '__main__':
    unittest.main()
