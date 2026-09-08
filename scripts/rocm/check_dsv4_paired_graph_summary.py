#!/usr/bin/env python3
"""Synthetic fixture tests for the ABBA artifact auditor (not GPU evidence)."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from summarize_dsv4_paired_graph_abba import summarize, summarize_fixed_processes


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

    def fixed_states(self):
        original = json.loads(self.state.read_text())
        paths = []
        for i, block in enumerate(original['blocks']):
            path = self.state.parent / f'process{i}.json'
            path.write_text(json.dumps(dict(
                status=original['status'], pid=100 + i, fixed_warmup_single=True,
                final_arm=block['candidate'], blocks=[dict(block, index=0)],
            )))
            paths.append(path)
        return paths

    def test_fixed_process_delta_and_identity(self):
        result = summarize_fixed_processes(self.fixed_states())
        self.assertIsNone(result['pid'])
        self.assertEqual(result['pids'], [100, 101, 102, 103])
        for values in result['metrics'].values():
            self.assertAlmostEqual(values['delta_pct'], 1)

    def test_fixed_process_rejects_wrong_arm_or_reused_pid(self):
        for field, bad in [('pid', 100), ('final_arm', True), ('fixed_warmup_single', False),
                           ('attention_issue_order', 3), ('ar_blocks', 4)]:
            paths = self.fixed_states()
            data = json.loads(paths[1].read_text())
            data[field] = bad
            paths[1].write_text(json.dumps(data))
            with self.subTest(field=field), self.assertRaises(AssertionError):
                summarize_fixed_processes(paths)

    def geometry_states(self):
        paths = self.fixed_states()
        for path, blocks in zip(paths, (4,16,16,4)):
            data = json.loads(path.read_text())
            data.update(ar_blocks=blocks, final_arm=True)
            data['blocks'][0]['candidate'] = True
            path.write_text(json.dumps(data))
        return paths

    def test_geometry_delta_uses_blocks_not_down_arm(self):
        result = summarize_fixed_processes(self.geometry_states(), ar_geometry=True)
        for values in result['metrics'].values():
            self.assertAlmostEqual(values['delta_pct'], 1)

    def test_geometry_rejects_wrong_grid_and_changed_down(self):
        for key, bad in [('ar_blocks',8),('final_arm',False)]:
            paths = self.geometry_states()
            data = json.loads(paths[1].read_text())
            data[key] = bad
            paths[1].write_text(json.dumps(data))
            with self.subTest(key=key), self.assertRaises(AssertionError):
                summarize_fixed_processes(paths, ar_geometry=True)


if __name__ == '__main__':
    unittest.main()
