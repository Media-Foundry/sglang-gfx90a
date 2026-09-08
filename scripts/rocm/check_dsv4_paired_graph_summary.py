#!/usr/bin/env python3
"""Synthetic fixture tests for the ABBA artifact auditor (not GPU evidence)."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from summarize_dsv4_paired_graph_abba import summarize


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        ids = list(range(256))
        c1sha = hashlib.sha256(struct.pack('<256I', *ids)).hexdigest()
        c32sha = hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()
        self.state = root / 'state.json'
        state = dict(status='complete_pending_c32_hash_review', final_arm=False,
                     pid=0, blocks=[])
        for i, enabled in enumerate([True, False, False, True]):
            factor = 1.01 if enabled else 1
            c1 = dict(status='complete', measurements=[dict(
                case=str(case), rep=rep, tokens=256, output_ids=ids,
                sha256=c1sha, tok_s=80 * factor)
                for case in range(3) for rep in range(4)])
            wave = dict(lengths=[256] * 32, finish_reasons=['length'] * 32,
                        output_ids=[ids] * 32, completion_sha256=[c32sha] * 32,
                        spec_accept_length_mean=None, aggregate_tok_s=1000 * factor,
                        resident_bs32_tok_s=1100 * factor)
            c32 = dict(round_count=6, request_count=32, tokens=256,
                       selected_workload_sha256='synthetic', rounds=[wave] * 6)
            block = dict(index=i, candidate=enabled, status='complete')
            for kind, data in [('c1', c1), ('c32', c32),
                               ('france', dict(exact_count=32, request_count=32))]:
                file = root / f'{i}.{kind}.json'
                file.write_text(json.dumps(data))
                block[kind] = str(file)
            state['blocks'].append(block)
        self.state.write_text(json.dumps(state))

    def test_known_delta(self):
        result = summarize(self.state)
        for values in result['metrics'].values():
            self.assertAlmostEqual(values['delta_pct'], 1)
        self.assertEqual(result['c32_completion_length_finish_hash_valid'], 768)

    def test_incomplete_state_rejected(self):
        data = json.loads(self.state.read_text())
        data['status'] = 'running'
        self.state.write_text(json.dumps(data))
        with self.assertRaises(AssertionError):
            summarize(self.state)

    def test_corrupt_completion_rejected(self):
        state = json.loads(self.state.read_text())
        file = Path(state['blocks'][1]['c32'])
        data = json.loads(file.read_text())
        data['rounds'][2]['output_ids'][0][0] = 900
        file.write_text(json.dumps(data))
        with self.assertRaises(AssertionError):
            summarize(self.state)

    def test_workload_change_rejected(self):
        state = json.loads(self.state.read_text())
        file = Path(state['blocks'][2]['c32'])
        data = json.loads(file.read_text())
        data['selected_workload_sha256'] = 'different'
        file.write_text(json.dumps(data))
        with self.assertRaises(AssertionError):
            summarize(self.state)


if __name__ == '__main__':
    unittest.main()
