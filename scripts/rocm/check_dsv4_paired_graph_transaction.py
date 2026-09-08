#!/usr/bin/env python3
"""CPU-only failure injection for the isolated paired-graph transaction."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

PATH = Path(__file__).resolve().parents[2] / (
    "python/sglang/srt/model_executor/runner_backend/paired_graph_transaction.py"
)
SPEC = importlib.util.spec_from_file_location("paired_graph_transaction", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TransactionTest(unittest.TestCase):
    def setUp(self):
        self.graph, self.output, self.c1, self.c1out = (object() for _ in range(4))
        self.backend = SimpleNamespace(
            _pool=(1, 0), _graphs={32: self.graph, 1: self.c1},
            _outputs={32: self.output, 1: self.c1out},
        )
        self.allocator_calls = []

    def run_capture(self, capture, pool=(2, 0), setter=None):
        return MODULE.capture_alternative(
            self.backend, 32, pool=pool,
            set_allocator_pool=setter or self.allocator_calls.append,
            capture=capture,
        )

    def assert_restored(self):
        self.assertIs(self.backend._graphs[32], self.graph)
        self.assertIs(self.backend._outputs[32], self.output)
        self.assertIs(self.backend._graphs[1], self.c1)
        self.assertIs(self.backend._outputs[1], self.c1out)
        self.assertEqual(self.backend._pool, (1, 0))
        self.assertEqual(self.allocator_calls[-1], (1, 0))

    def test_success_retains_candidate_pair_and_restores_baseline(self):
        graph, output = object(), object()
        def capture():
            self.assertEqual(self.backend._pool, (2, 0))
            self.backend._graphs[32] = graph
            self.backend._outputs[32] = output
        result = self.run_capture(capture)
        self.assertIs(result.graph, graph)
        self.assertIs(result.output, output)
        self.assertEqual(result.pool, (2, 0))
        self.assert_restored()

    def test_failure_at_each_capture_stage(self):
        for writes in range(3):
            with self.subTest(writes=writes):
                def capture():
                    if writes >= 1:
                        self.backend._graphs[32] = object()
                    if writes >= 2:
                        self.backend._outputs[32] = object()
                    raise RuntimeError("injected capture/upload failure")
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    self.run_capture(capture)
                self.assert_restored()

    def test_allocator_setup_failure_restores(self):
        def setter(pool):
            self.allocator_calls.append(pool)
            if pool == (2, 0):
                raise RuntimeError("allocator setup")
        with self.assertRaisesRegex(RuntimeError, "allocator setup"):
            self.run_capture(lambda: None, setter=setter)
        self.assert_restored()

    def test_no_capture_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "did not replace"):
            self.run_capture(lambda: None)
        self.assert_restored()

    def test_output_reuse_rejected(self):
        def capture():
            self.backend._graphs[32] = object()
        with self.assertRaisesRegex(RuntimeError, "output object"):
            self.run_capture(capture)
        self.assert_restored()

    def test_same_or_missing_pool_rejected_before_capture(self):
        for pool in (None, (1, 0)):
            with self.assertRaises(ValueError):
                self.run_capture(lambda: self.fail("must not capture"), pool=pool)
        self.assertEqual(self.allocator_calls, [])

    def test_missing_baseline_rejected_before_allocator(self):
        del self.backend._outputs[32]
        with self.assertRaises(KeyError):
            self.run_capture(lambda: self.fail("must not capture"))
        self.assertEqual(self.allocator_calls, [])


if __name__ == "__main__":
    unittest.main()
