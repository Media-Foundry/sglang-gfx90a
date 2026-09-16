"""CPU metadata equivalence and fail-closed service call-site contracts."""
import ast
from pathlib import Path
import unittest
import os
import subprocess

import numpy as np
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_owner import host_plan


class OwnerTest(unittest.TestCase):
    def test_prefix_ragged_and_roundrobin(self):
        for ext,pre in [([8192]*4,[0]*4),([8192,8192,8191,8192],[0]*4),
                        ([1,15,17],[2047,2048,8191]),([2048],[0]),([32768],[128]),
                        ([7,5],[0,0])]:
            allrows=[]
            expected=np.concatenate([(np.arange(1,n+1)+p)//4 for n,p in zip(ext,pre)])
            reference=[list(range(i,min(i+16,len(expected)))) for i in range(0,len(expected),16)
                       if any(expected[i:i+16]>512)]
            for rank in range(8):
                lengths,ids,valid,inverse=host_plan(ext,pre,rank)
                np.testing.assert_array_equal(lengths,expected)
                selected=[j for group in reference[rank::8] for j in group]
                np.testing.assert_array_equal(ids[valid.astype(bool)],selected)
                self.assertEqual(len(ids)%16,0)
                allrows.extend(ids[valid.astype(bool)].tolist())
                for slot,row in enumerate(ids):
                    if valid[slot]:self.assertEqual(inverse[row],rank*len(ids)+slot)
            self.assertEqual(len(allrows),len(set(allrows)))
            self.assertTrue(set(np.flatnonzero(expected>512)).issubset(allrows))

    def test_invalid_metadata(self):
        for ext,pre,rank in [([],[],0),([1],[],0),([0],[0],0),([1],[-1],0),([1],[0],8)]:
            with self.assertRaises(ValueError):host_plan(ext,pre,rank)

    def test_guard_requires_explicit_native_scope(self):
        root=Path(__file__).resolve().parents[4]
        tree=ast.parse((root/'python/sglang/srt/layers/attention/dsv4/indexer.py').read_text())
        branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                    and 'SGLANG_DSV4_C4_PREFILL_QUERY_OWNER' in ast.unparse(n.test))
        condition=ast.unparse(branch.test)
        for term in ("'0'",'not use_fp4_indexer','not enable_multi_stream',
                     '_use_c4_prefill_query_reuse','attn_dcp_size','spec_algorithm.is_none()',
                     'compressor.ratio == 4','not indexer_metadata.use_prefill_cuda_graph',
                     'not self.debug_use_external_c4_sparse_indices','self.hisparse_coordinator is None',
                     'get_global_indexer_capturer() is None','SGLANG_OPT_USE_TOPK_V2',
                     'SGLANG_DSV4_C4_PREFILL_QUERY_RUNTIME_M'):
            self.assertIn(term,condition)

    def test_launcher_default_scope(self):
        root=Path(__file__).resolve().parents[4]
        prefix=(root/'scripts/rocm_dsv4_flash.sh').read_text().split('\n# One TP4 replica')[0]
        self.assertTrue(prefix.endswith('fi\n'))
        def flag(values):
            script=prefix+'\nprintf "%s" "${SGLANG_DSV4_C4_PREFILL_QUERY_OWNER-unset}"\n'
            return subprocess.check_output(['bash','-c',script],env={'PATH':os.defpath,**values},text=True)
        tp8={'TP_SIZE':'8','SGLANG_DSV4_GFX90A_TP8_MULTI_REQUEST_PROFILE':'1'}
        both={**tp8,'SGLANG_DSV4_GFX90A_PREFILL_THROUGHPUT_PROFILE':'1'}
        self.assertEqual(flag({}),'unset')
        self.assertEqual(flag(tp8),'unset')
        self.assertEqual(flag(both),'1')
        self.assertEqual(flag({**both,'SGLANG_DSV4_C4_PREFILL_QUERY_OWNER':'0'}),'0')
        self.assertEqual(flag({**both,'TP_SIZE':'4'}),'unset')

    def test_measured_narrow_body_preserved(self):
        root=Path(__file__).resolve().parents[4]
        measured=ast.parse((root/'.agents/experiments/dsv4_c16_indexer_owner_service_20260915/tested_helper.py').read_text())
        current=ast.parse((root/'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_owner.py').read_text())
        # Diagnostics and default-off K32 must not change the measured fallback.
        current.body=[n for n in current.body if not (
            isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_detail_mark' for t in n.targets)
            or isinstance(n,ast.If) and 'SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR' in ast.unparse(n.test))]
        fn=next(n for n in current.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        k32=[n for n in fn.body if isinstance(n,ast.If) and 'SGLANG_DSV4_C4_PREFILL_OWNER_K32' in ast.unparse(n.test)]
        self.assertEqual(len(k32),1)
        self.assertIn("'0'",ast.unparse(k32[0].test))
        stripped=[]
        for n in fn.body:
            if n is k32[0]:continue
            if isinstance(n,ast.Assign) and ast.unparse(n)=='scores = None':continue
            if isinstance(n,ast.If) and ast.unparse(n.test)=='scores is None':
                self.assertEqual(len(n.body),1)
                self.assertTrue(ast.unparse(n.body[0]).startswith('scores = prefill_query_reuse4('))
                stripped.extend(n.body)
            elif isinstance(n,ast.If) and ast.unparse(n.test)=='_detail_mark is not None':
                self.assertEqual(len(n.body),1)
                self.assertIn(n.body[0].value.args[0].value,range(55,62))
            else:stripped.append(n)
        fn.body=stripped
        guards=[n for n in fn.body if isinstance(n,ast.If) and 'not ext or max(' in ast.unparse(n.test)]
        self.assertEqual(len(guards),1)
        fn.body.remove(guards[0])
        # Strip only the opt-in wide plumbing. The previously measured narrow
        # packing, scoring, integer gather and reconstruction must stay intact.
        fn.body=[n for n in fn.body if not (isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id in ('wide','owner_kw') for t in n.targets))]
        class Narrow(ast.NodeTransformer):
            def visit_IfExp(self,n):
                return self.visit(n.orelse) if isinstance(n.test,ast.Name) and n.test.id=='wide' else self.generic_visit(n)
            def visit_If(self,n):
                return None if isinstance(n.test,ast.Name) and n.test.id=='wide' else self.generic_visit(n)
            def visit_Call(self,n):
                if isinstance(n.func,ast.Name) and n.func.id=='prefill_query_reuse4':
                    n.keywords=[k for k in n.keywords if not (k.arg is None and
                        (isinstance(k.value,ast.IfExp) or isinstance(k.value,ast.Name) and k.value.id=='owner_kw'))]
                return self.generic_visit(n)
        Narrow().visit(current)
        self.assertEqual(ast.dump(current),ast.dump(measured))

    def test_wide_global_and_local_admission(self):
        from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_query_reuse import _wide_rows_supported
        for local,global_m in [(3584,32768),(3840,32767),(4096,32768),(16,8192),(8192,65536)]:
            self.assertTrue(_wide_rows_supported(local,global_m))
        for local,global_m in [(0,32768),(4097,32768),(4112,32768),(3584,8191),(16,65537),(15,32768)]:
            self.assertFalse(_wide_rows_supported(local,global_m))
        for m in (8192,32767,65536):self.assertTrue(_wide_rows_supported(m))
        for m in (0,3584,8191,65537):self.assertFalse(_wide_rows_supported(m))


if __name__=='__main__':unittest.main()
