"""CPU-only benchmark contract tests; never import the model or initialize HIP."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / 'scripts/rocm'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('open_code_bench_test', SCRIPTS / 'bench_dsv4_open_code_decode.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
audit_spec = importlib.util.spec_from_file_location('open_code_audit_test', SCRIPTS / 'summarize_dsv4_open_code_matrix.py')
audit = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(audit)


class TestOpenCodeMatrix(unittest.TestCase):
    def run_fake(self, count=64, empty=False):
        with tempfile.TemporaryDirectory() as folder:
            manifest = Path(folder) / 'inputs.json'
            output = Path(folder) / 'out.json'
            manifest.write_text(json.dumps({'requests': [
                {'index': i, 'input_ids': [i+100]} for i in range(count)]}))
            def stream(url, payload, timeout):
                ids = [] if empty else [10, 20, 30]
                return {'output_ids': ids, 'text': 'example code',
                        'meta_info': {'completion_tokens': len(ids),
                                      'finish_reason': {'type': 'stop'}}}, [(10., 1), (11., 2), (12., 3)]
            argv = ['bench', '--base-url', 'http://unused', '--inputs', str(manifest),
                    '--request-count', str(count), '--rounds', '1', '--seconds', '0',
                    '--tokens', '3', '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(bench, 'post_stream', stream), contextlib.redirect_stdout(io.StringIO()):
                bench.main()
            return json.loads(output.read_text())

    def test_c64_real_distinct_rows_and_raw_timestamps(self):
        result = self.run_fake()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['median_decode_tok_s'], 64.)
        wave = result['rounds'][0]['waves'][0]
        self.assertEqual(len({r['index'] for r in wave['requests']}), 64)
        self.assertEqual(wave['resident_tokens'], 128)
        self.assertTrue(all(len(r['samples']) == 3 for r in wave['requests']))
        self.assertGreater(wave['http_wall_seconds'], 0)
        self.assertFalse(result['ignore_eos'])

    def test_c1_keeps_original_window_formula(self):
        self.assertEqual(self.run_fake(count=1)['median_decode_tok_s'], 1.)

    def test_empty_output_cannot_pass(self):
        with self.assertRaises(AssertionError):
            self.run_fake(empty=True)

    def test_auditor_recomputes_and_rejects_counter_corruption(self):
        data = self.run_fake(count=1)
        self.assertEqual(audit.decode_summary(data)['median'], 1.)
        data['rounds'][0]['waves'][0]['resident_tokens'] += 1
        with self.assertRaises(AssertionError):
            audit.decode_summary(data)

    def test_auditor_rejects_false_prefill_and_cache_hits(self):
        data = dict(rounds=[dict(completion_lengths=[1], request_count=1,
            cached_tokens=[0], total_prompt_tokens=8, prefill_wall_s=2.,
            aggregate_input_tok_s=4.)], median_input_tok_s=4., input_manifest_sha256='fixture')
        self.assertEqual(audit.prefill_summary(data)['median'], 4.)
        data['rounds'][0]['cached_tokens'] = [1]
        with self.assertRaises(AssertionError):
            audit.prefill_summary(data)


if __name__ == '__main__':
    unittest.main()
