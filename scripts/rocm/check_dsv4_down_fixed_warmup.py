#!/usr/bin/env python3
"""CPU sequencing/failure tests, not GPU numerical correctness evidence."""

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'python/sglang/srt/model_executor/runner_backend/dsv4_down_fixed_warmup.py'
SPEC = importlib.util.spec_from_file_location('fixed_warmup', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class WarmupTest(unittest.TestCase):
    def setUp(self):
        self.events, self.arm, self.raw = [], None, False
        self.free = iter([5 * 1024**3, 4 * 1024**3, 4 * 1024**3 - 4 * 1024**2])
        self.key = SimpleNamespace(size=32, dsa_variant='dense', stream_idx=None, variant_label=None)
        self.device = SimpleNamespace(
            synchronize=lambda: self.events.append('sync'),
            mem_get_info=lambda: (next(self.free), 64 * 1024**3),
        )
        self.group = SimpleNamespace(barrier=lambda: self.events.append('barrier'))
        MODULE.memory_consensus = lambda group, valid: valid

    @contextmanager
    def override(self, arm):
        old, self.arm = self.arm, arm
        try:
            yield
        finally:
            self.arm = old

    def reset(self):
        self.events.append('reset')
        self.raw = True

    def forward(self):
        self.assertTrue(self.raw)
        self.raw = False
        self.events.append(('forward', self.arm))

    def run_pair(self, reset=True, forward=None):
        return MODULE.warmup_pair(
            self.key, forward or self.forward, self.reset if reset else None,
            device=self.device, group=self.group, override=self.override,
        )

    def test_order_drain_and_metadata_reset(self):
        self.assertTrue(self.run_pair())
        self.assertEqual(self.events, [
            'sync', 'barrier', 'reset', 'barrier', ('forward', False),
            'sync', 'reset', 'reset', 'barrier', ('forward', True),
            'sync', 'reset', 'barrier',
        ])
        self.assertIsNone(self.arm)
        self.assertTrue(self.raw)

    def test_other_tiers_and_sparse_untouched(self):
        for size, variant in ((1, None), (2, 'dense'), (64, 'dense'), (32, 'sparse')):
            self.key.size, self.key.dsa_variant = size, variant
            self.assertFalse(self.run_pair())
        self.assertEqual(self.events, [])

    def test_invalid_key_or_missing_hook_rejected(self):
        with self.assertRaises(RuntimeError):
            self.run_pair(reset=False)
        self.key.stream_idx = 0
        with self.assertRaises(RuntimeError):
            self.run_pair()
        self.assertEqual(self.events, [])

    def test_failure_restores_selector_without_retry(self):
        def fail():
            self.forward()
            raise RuntimeError('injected')
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            self.run_pair(forward=fail)
        self.assertIsNone(self.arm)
        self.assertEqual([e for e in self.events if isinstance(e, tuple)], [('forward', False)])

    def test_remote_memory_rejection_prevents_forward(self):
        MODULE.memory_consensus = lambda group, valid: False
        with self.assertRaisesRegex(RuntimeError, 'admission'):
            self.run_pair()
        self.assertNotIn(('forward', False), self.events)

    def test_post_memory_limit_fails_closed(self):
        self.free = iter([5 * 1024**3, 4 * 1024**3, 3 * 1024**3])
        with self.assertRaisesRegex(RuntimeError, 'exceeded'):
            self.run_pair()
        self.assertIsNone(self.arm)

    def test_runner_scope_guards(self):
        args = SimpleNamespace(max_total_tokens=1048576, pp_size=1, dp_size=1,
                               enable_dp_attention=False)
        model = SimpleNamespace(server_args=args, device=0, is_draft_worker=False,
                                spec_algorithm=SimpleNamespace(is_none=lambda: True),
                                model_config=SimpleNamespace(hf_config=object()))
        runner = type('DecodeCudaGraphRunner', (), {})()
        runner.__dict__.update(model_runner=model, enable_torch_compile=False,
                               ragged_verify_mode=False, enable_profile_cuda_graph=False,
                               capture_bs=[1, 32])
        backend = SimpleNamespace(_cuda_graph_runner=runner,
                                  _tp_group=SimpleNamespace(world_size=8),
                                  _down_graph_pair=None)
        fake_torch = SimpleNamespace(version=SimpleNamespace(hip=True),
            cuda=SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(gcnArchName='gfx90a')))
        modules = {
            'torch': fake_torch,
            'sglang.srt.configs.model_config': SimpleNamespace(is_deepseek_v4=lambda _: True),
            'sglang.srt.distributed': SimpleNamespace(get_moe_expert_parallel_world_size=lambda: 1),
        }
        with patch.dict('sys.modules', modules):
            MODULE.validate_runner(backend, memory_saver=False)
            for owner, field, bad in (
                (args, 'max_total_tokens', 131072), (args, 'dp_size', 2),
                (args, 'pp_size', 2), (args, 'enable_dp_attention', True),
                (model, 'is_draft_worker', True),
                (model, 'spec_algorithm', SimpleNamespace(is_none=lambda: False)),
                (runner, 'capture_bs', [1, 16]), (runner, 'enable_torch_compile', True),
                (runner, 'ragged_verify_mode', True), (runner, 'enable_profile_cuda_graph', True),
                (backend._tp_group, 'world_size', 4), (backend, '_down_graph_pair', object()),
            ):
                old = getattr(owner, field)
                setattr(owner, field, bad)
                with self.subTest(field=field), self.assertRaises(RuntimeError):
                    MODULE.validate_runner(backend, memory_saver=False)
                setattr(owner, field, old)
            with self.assertRaises(RuntimeError):
                MODULE.validate_runner(backend, memory_saver=True)


if __name__ == '__main__':
    unittest.main()
