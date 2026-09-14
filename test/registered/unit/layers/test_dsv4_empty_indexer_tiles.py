"""CPU-only selector and wrapper contracts; GPU oracle checks score bits."""
import ast
from contextlib import redirect_stdout
from enum import Enum
from pathlib import Path
import os
import io
import subprocess
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch


class TestEmptyIndexerTiles(unittest.TestCase):
    def test_query_reuse_hit_log_identifies_actual_rank(self):
        root=Path(__file__).resolve().parents[4]
        tree=ast.parse((root/'python/sglang/srt/layers/attention/dsv4/indexer.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Name) and n.func.id=='print'
               and any(isinstance(v,ast.Constant) and isinstance(v.value,str)
                       and 'prefill query-reuse4 selected' in v.value for v in ast.walk(n))]
        self.assertEqual(len(calls),1)
        expression=compile(ast.Expression(calls[0]),'<actual query hit log>','eval')
        for rank in range(8):
            out=io.StringIO()
            with redirect_stdout(out):
                eval(expression,dict(get_parallel=lambda:NS(tp_rank=rank),
                                     q=NS(shape=(32767,1,64,128)),
                                     indexer_metadata=NS(max_c4_seq_len=2048)))
            self.assertIn(f'TP{rank}]',out.getvalue())
            self.assertIn('prefill query-reuse4 selected: rows=32767 C4_capacity=2048',out.getvalue())

    def setUp(self):
        root = Path(__file__).resolve().parents[4]
        tree = ast.parse((root/'python/sglang/srt/layers/attention/dsv4/indexer.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'C4IndexerBackendMixin')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_use_c4_empty_decode_tiles')
        prefill = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_use_c4_empty_prefill_tiles')
        query_reuse = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_use_c4_prefill_query_reuse')
        wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fp8_paged_mqa_logits_torch')
        self.mode = Enum('Mode', 'DECODE EXTEND TARGET_VERIFY IDLE DRAFT_EXTEND')
        self.flag = False
        self.hip = self.gfx90a = True
        self.capturing = False
        self.ps = NS(tp_size=8,attn_tp_size=8,moe_ep_size=1,attn_cp_size=1,pp_size=1)
        self.namespace = dict(
            envs=NS(SGLANG_DSV4_GFX90A_AR_INDEXER_EMPTY_TILE_SKIP=NS(get=lambda: self.flag)),
            is_hip=lambda: self.hip, is_gfx90a_supported=lambda: self.gfx90a,
            get_parallel=lambda:self.ps,
            ForwardMode=self.mode, torch=NS(Tensor=object,cuda=NS(is_current_stream_capturing=lambda:self.capturing)), Any=object, os=os)
        exec(compile(ast.Module(body=[method, prefill, query_reuse, wrapper], type_ignores=[]), '<indexer contract>', 'exec'), self.namespace)
        self.backend = NS(token_to_kv_pool=NS(_unified_kv=True), is_draft_worker=False,
                          is_dspark_target=False, mtp_enabled=False,
                          dsa_topk_backend=NS(is_sgl_kernel=lambda:True))
        self.backend._use_c4_empty_prefill_tiles=lambda fb,idx:self.namespace['_use_c4_empty_prefill_tiles'](self.backend,fb,idx)

    def test_query_reuse_requires_native_tp8_prefill_and_explicit_opt_in(self):
        idx=NS(compressor=NS(_debug_original_v4=True),index_topk=512)
        fb=NS(forward_mode=self.mode.EXTEND)
        def call(rows=32768):
            return self.namespace['_use_c4_prefill_query_reuse'](self.backend,fb,idx,rows)
        flags=dict(SGLANG_DSV4_C4_PREFILL_QUERY_REUSE4='1',
                   SGLANG_DSV4_C4_TRIVIAL_LOGITS_SKIP='1',
                   SGLANG_DSV4_C4_PREFILL_EMPTY_TILE_SKIP='1',
                   SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER='3')
        with patch.dict(os.environ,flags):
            self.assertTrue(call())
            for flag in flags:
                with patch.dict(os.environ,{flag:'0'}):self.assertFalse(call())
            for rows in (0,128,8191,65537):self.assertFalse(call(rows))
            for rows in (8192,32767,65536):self.assertTrue(call(rows))
            for mode in self.mode:
                fb.forward_mode=mode
                self.assertEqual(call(),mode==self.mode.EXTEND)
            fb.forward_mode=self.mode.EXTEND
            for name,value in [('tp_size',4),('attn_tp_size',4),('moe_ep_size',2),('attn_cp_size',2),('pp_size',2)]:
                old=getattr(self.ps,name);setattr(self.ps,name,value)
                self.assertFalse(call());setattr(self.ps,name,old)
            self.capturing=True;self.assertFalse(call());self.capturing=False
            fb.tbo_parent_token_range=(0,32768);self.assertFalse(call())
            fb.tbo_parent_token_range=None
            fb._original_forward_mode=self.mode.TARGET_VERIFY;self.assertFalse(call())

    def test_prefill_is_independent_and_fail_closed(self):
        indexer=NS(compressor=NS(_debug_original_v4=True),index_topk=512)
        def call(mode):
            return self.namespace['_use_c4_empty_prefill_tiles'](self.backend,NS(forward_mode=mode),indexer)
        flag='SGLANG_DSV4_C4_PREFILL_EMPTY_TILE_SKIP'
        with patch.dict(os.environ,{flag:'0'}):
            self.assertFalse(call(self.mode.EXTEND))
        with patch.dict(os.environ,{flag:'1','SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER':'3'}):
            self.assertTrue(call(self.mode.EXTEND))
            for mode in self.mode:
                if mode!=self.mode.EXTEND:self.assertFalse(call(mode))
            for role in ('is_draft_worker','is_dspark_target','mtp_enabled'):
                setattr(self.backend,role,True);self.assertFalse(call(self.mode.EXTEND))
                delattr(self.backend,role);self.assertFalse(call(self.mode.EXTEND))
                setattr(self.backend,role,False)
            indexer.compressor._debug_original_v4=False
            self.assertFalse(call(self.mode.EXTEND))
            del indexer.compressor._debug_original_v4
            self.assertFalse(call(self.mode.EXTEND))
            indexer.compressor._debug_original_v4=True
            indexer.index_topk=256;self.assertFalse(call(self.mode.EXTEND))
            indexer.index_topk=512
            self.backend.token_to_kv_pool._unified_kv=False
            self.assertFalse(call(self.mode.EXTEND))

    def selected(self, mode=None):
        return self.namespace['_use_c4_empty_decode_tiles'](
            self.backend, NS(forward_mode=mode or self.mode.DECODE))

    def test_default_off_and_native_hit(self):
        self.assertFalse(self.selected())
        self.flag = True
        self.assertTrue(self.selected())

    def test_excludes_non_decode(self):
        self.flag = True
        for mode in self.mode:
            if mode != self.mode.DECODE:
                with self.subTest(mode=mode):
                    self.assertFalse(self.selected(mode))

    def test_excludes_wrong_arch(self):
        self.flag = True
        self.hip = False
        self.assertFalse(self.selected())
        self.hip, self.gfx90a = True, False
        self.assertFalse(self.selected())

    def test_excludes_speculative_and_paged_pool(self):
        self.flag = True
        for role in ('is_draft_worker', 'is_dspark_target', 'mtp_enabled'):
            with self.subTest(role=role):
                setattr(self.backend, role, True)
                self.assertFalse(self.selected())
                setattr(self.backend, role, False)
        self.backend.token_to_kv_pool._unified_kv = False
        self.assertFalse(self.selected())

    def test_unknown_role_fails_closed(self):
        self.flag = True
        for role in ('is_draft_worker', 'is_dspark_target', 'mtp_enabled'):
            with self.subTest(role=role):
                delattr(self.backend, role)
                self.assertFalse(self.selected())
                setattr(self.backend, role, False)

    def test_public_wrapper_forwards_flag(self):
        calls = []
        result = object()
        def helper(*args, **kwargs):
            calls.append(kwargs)
            return result
        self.namespace['_fp8_paged_mqa_logits_triton'] = helper
        for flag in (False, True):
            observed = self.namespace['fp8_paged_mqa_logits_torch'](
                NS(shape=(1,1,64,128)), NS(shape=(10,64,1,132)), NS(shape=(1,64)),
                NS(shape=(1,)), NS(shape=(1,4096)), None, 262144, False,
                skip_empty_tiles=flag)
            self.assertIs(observed, result)
            self.assertEqual(calls[-1], dict(skip_trivial_topk=0, skip_empty_tiles=flag))

    def test_launcher_default_and_explicit_override(self):
        # Evaluate only the profile assignment block, never source the launcher
        # or invoke any service/GPU commands in this CPU contract test.
        root = Path(__file__).resolve().parents[4]
        source = (root/'scripts/rocm_dsv4_flash.sh').read_text()
        start = source.index('GFX90A_TP8_MULTI_REQUEST_PROFILE="${SGLANG_DSV4_GFX90A_TP8_MULTI_REQUEST_PROFILE:-0}"')
        end = source.index('\nfi\n', start)+4
        block = source[start:end]
        flag = 'SGLANG_DSV4_GFX90A_AR_INDEXER_EMPTY_TILE_SKIP'
        for profile, override, tp, expected in (
            ('1',None,'8','1'), ('1','0','8','0'), ('1','1','8','1'),
            ('0',None,'8','unset'), ('1',None,'4','unset')):
            with self.subTest(profile=profile, override=override, tp=tp):
                env = dict(PATH=os.environ['PATH'], TP_SIZE=tp,
                           SGLANG_DSV4_GFX90A_TP8_MULTI_REQUEST_PROFILE=profile)
                if override is not None:
                    env[flag] = override
                result = subprocess.run(['bash','--noprofile','--norc','-c',
                    'set -eu\n'+block+'\nprintf "%s" "${'+flag+'-unset}"'],
                    env=env, text=True, capture_output=True, check=True)
                self.assertEqual(result.stdout, expected)

    def test_prefill_launcher_requires_both_profiles_and_preserves_override(self):
        self._check_prefill_launcher_flag('SGLANG_DSV4_C4_PREFILL_EMPTY_TILE_SKIP')

    def test_query_reuse_launcher_requires_both_profiles_and_preserves_override(self):
        self._check_prefill_launcher_flag('SGLANG_DSV4_C4_PREFILL_QUERY_REUSE4')

    def _check_prefill_launcher_flag(self, flag):
        root = Path(__file__).resolve().parents[4]
        source = (root/'scripts/rocm_dsv4_flash.sh').read_text()
        start = source.index('GFX90A_TP8_MULTI_REQUEST_PROFILE="${SGLANG_DSV4_GFX90A_TP8_MULTI_REQUEST_PROFILE:-0}"')
        end = source.index('\nfi\n', start)+4
        block = source[start:end]
        for tp_profile,prefill,tp,ep,a2a,override,expected in (
            ('1','1','8','1','none',None,'1'),
            ('1','1','8','1','none','0','0'),
            ('1','1','8','1','none','1','1'),
            ('1','0','8','1','none',None,'unset'),
            ('0','1','8','1','none',None,'unset'),
            ('1','1','4','1','none',None,'unset'),
            ('1','1','8','2','none',None,'unset'),
            ('1','1','8','1','mori',None,'unset')):
            env=dict(PATH=os.environ['PATH'],TP_SIZE=tp,EP_SIZE=ep,MOE_A2A_BACKEND=a2a,
                     SGLANG_DSV4_GFX90A_TP8_MULTI_REQUEST_PROFILE=tp_profile,
                     GFX90A_PREFILL_THROUGHPUT_PROFILE=prefill)
            if override is not None:env[flag]=override
            with self.subTest(env=env):
                result=subprocess.run(['bash','--noprofile','--norc','-c',
                    'set -eu\n'+block+'\nprintf "%s" "${'+flag+'-unset}"'],
                    env=env,text=True,capture_output=True,check=True)
                self.assertEqual(result.stdout,expected)


if __name__ == '__main__':
    unittest.main()
