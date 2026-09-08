#!/usr/bin/env python3
"""CPU tests of actual pair methods with mocked device/capture/collectives."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from check_dsv4_paired_graph_transaction import MODULE as transaction

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'python/sglang/srt/model_executor/runner_backend/dsv4_down_graph_pair.py'
tree = ast.parse(PATH.read_text())
definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
ns = dict(capture_alternative=transaction.capture_alternative,
          down_uniform_capture=lambda arm: nullcontext(),
          logger=NS(info=lambda *a: None, warning=lambda *a: None), _pair=None)
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(PATH), 'exec'), ns)


class PairTest(unittest.TestCase):
    def test_numeric_control_wire_contract(self):
        for value in (0, 1):
            self.assertIs(ns['parse_arm_request']({'dsv4_down_uniform_arm': value}), bool(value))
        for value in (-1, 2, 0.0, '1', None, (False,)):
            with self.assertRaises(ValueError):
                ns['parse_arm_request']({'dsv4_down_uniform_arm': value})
        with self.assertRaises(ValueError):
            ns['parse_arm_request']({'dsv4_down_uniform_arm': 1, 'another': 0})

    def setUp(self):
        self.key = NS(size=32, stream_idx=None, variant_label=None, dsa_variant=None)
        # Hashable shape-key surrogate, with the same public fields.
        self.key = type('Key', (), vars(self.key))()
        self.c1 = type('Key', (), dict(size=1))()
        self.graph, self.output = object(), object()
        self.free = iter([8 * 1024**3] + [8 * 1024**3 - 100] * 10)
        self.events = []
        ns['set_graph_pool_id'] = lambda p: self.events.append(('pool', p))
        self.backend = NS(
            _pool=(1, 0), _graphs={}, _outputs={},
            _device_module=NS(
                synchronize=lambda: self.events.append('sync'),
                mem_get_info=lambda: (next(self.free), 64 * 1024**3),
                graph_pool_handle=lambda: (2, 0)),
            _tp_group=NS(barrier=lambda: self.events.append('barrier')),
        )
        self.pair = ns['DownGraphPair'].__new__(ns['DownGraphPair'])
        self.pair.backend = self.backend
        self.pair.key = self.pair.alternative = None
        self.pair.enabled = False
        self.pair.consensus = lambda v: bool(v)
        ns['_pair'] = self.pair

    def capture(self, key=None):
        key = key or self.key
        def callback():
            self.backend._graphs[key] = object()
            self.backend._outputs[key] = object()
        self.pair.capture(key, callback, reset_after_capture=lambda: None)

    def test_raw_metadata_restored_before_second_warmup(self):
        state = {'kind': 'raw', 'captures': 0}
        def callback():
            self.assertEqual(state['kind'], 'raw', 'unexecuted graph metadata reused')
            self.backend._graphs[self.key] = object()
            self.backend._outputs[self.key] = object()
            state.update(kind='full_recorded_not_executed', captures=state['captures'] + 1)
        def reset():
            self.assertEqual(state['kind'], 'full_recorded_not_executed')
            state['kind'] = 'raw'
        self.pair.capture(self.key, callback, reset_after_capture=reset)
        self.assertEqual(state['captures'], 2)

    def test_missing_metadata_reset_fails_before_candidate(self):
        calls = []
        with self.assertRaisesRegex(RuntimeError, 'reset hook'):
            self.pair.capture(self.key, lambda: calls.append(1), reset_after_capture=None)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(self.pair.alternative)

    def test_capture_select_and_idle_switch(self):
        self.capture(self.c1)
        c1graph = self.backend._graphs[self.c1]
        self.capture()
        baseline = self.backend._graphs[self.key]
        self.assertIsNone(self.pair.selected(self.key))
        self.assertFalse(self.pair.switch(True, idle=False))
        self.assertTrue(self.pair.switch(True, idle=True))
        self.assertIs(self.pair.selected(self.key), self.pair.alternative)
        self.assertIsNone(self.pair.selected(self.c1))
        self.assertIs(self.backend._graphs[self.c1], c1graph)
        self.assertIs(self.backend._graphs[self.key], baseline)
        self.assertTrue(self.pair.switch(False, idle=True))
        self.assertIsNone(self.pair.selected(self.key))

    def test_remote_busy_rejects_without_barrier(self):
        self.capture()
        self.pair.consensus = lambda v: False
        self.events.clear()
        self.assertFalse(self.pair.switch(True, idle=True))
        self.assertEqual(self.events, [])

    def test_existing_dsa_dense_sparse_graphs(self):
        self.key.dsa_variant = 'dense'
        self.capture()
        sparse = type('Key', (), dict(size=32, stream_idx=None,
                                     variant_label=None, dsa_variant='sparse'))()
        self.capture(sparse)
        self.assertTrue(self.pair.switch(True, idle=True))
        self.assertIsNotNone(self.pair.selected(self.key))
        self.assertIsNone(self.pair.selected(sparse))
        self.assertIn(sparse, self.backend._graphs)

    def test_pre_capture_memory_rejection(self):
        self.free = iter([2 * 1024**3])
        with self.assertRaisesRegex(RuntimeError, 'admission'):
            self.capture()
        self.assertIsNone(self.pair.alternative)
        self.assertEqual(self.backend._pool, (1, 0))

    def test_post_capture_memory_rejection(self):
        self.free = iter([8 * 1024**3, 6 * 1024**3])
        with self.assertRaisesRegex(RuntimeError, 'exceeded'):
            self.capture()
        self.assertIsNone(self.pair.alternative)
        self.assertEqual(self.backend._pool, (1, 0))

    def test_invalid_arm_and_not_ready(self):
        self.assertFalse(self.pair.switch(True, idle=True))
        self.capture()
        for value in (1, 'true', None, (False,)):
            self.assertFalse(self.pair.switch(value, idle=True))

    def test_late_registration_memory_rejects_switch(self):
        self.capture()
        self.free = iter([6 * 1024**3])
        self.events.clear()
        self.assertFalse(self.pair.switch(True, idle=True))
        self.assertFalse(self.pair.enabled)
        self.assertNotIn('barrier', self.events)

    def test_cleanup_unregisters(self):
        self.capture()
        self.pair.cleanup()
        self.assertIsNone(ns['_pair'])
        self.assertIsNone(self.pair.alternative)
        self.assertFalse(ns['switch_arm'](True, idle=True))


if __name__ == '__main__':
    unittest.main()
