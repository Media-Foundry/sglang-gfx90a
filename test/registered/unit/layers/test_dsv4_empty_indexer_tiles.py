"""CPU-only selector and wrapper contracts; GPU oracle checks score bits."""
import ast
from enum import Enum
from pathlib import Path
import os
import subprocess
from types import SimpleNamespace as NS
import unittest


class TestEmptyIndexerTiles(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[4]
        tree = ast.parse((root/'python/sglang/srt/layers/attention/dsv4/indexer.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'C4IndexerBackendMixin')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_use_c4_empty_decode_tiles')
        wrapper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fp8_paged_mqa_logits_torch')
        self.mode = Enum('Mode', 'DECODE EXTEND TARGET_VERIFY IDLE DRAFT_EXTEND')
        self.flag = False
        self.hip = self.gfx90a = True
        self.namespace = dict(
            envs=NS(SGLANG_DSV4_GFX90A_AR_INDEXER_EMPTY_TILE_SKIP=NS(get=lambda: self.flag)),
            is_hip=lambda: self.hip, is_gfx90a_supported=lambda: self.gfx90a,
            ForwardMode=self.mode, torch=NS(Tensor=object), Any=object)
        exec(compile(ast.Module(body=[method, wrapper], type_ignores=[]), '<indexer contract>', 'exec'), self.namespace)
        self.backend = NS(token_to_kv_pool=NS(_unified_kv=True), is_draft_worker=False,
                          is_dspark_target=False, mtp_enabled=False)

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


if __name__ == '__main__':
    unittest.main()
