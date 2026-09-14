import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('mhc20_analysis',Path(__file__).with_name('analyze.py'))
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


class TestAnalysis(unittest.TestCase):
    def fixture(self):
        manifest={'requests':[{'input_ids':[i,100]} for i in range(16)]}
        responses=[dict(meta_info=dict(id=f'mhc20-B-0-{i}',completion_tokens=128,cached_tokens=0),
                        prompt_token_ids=[i,100],output_ids=[7]*128,text='valid') for i in range(16)]
        tokenizer=type('Tokenizer',(),{'decode':lambda self,*a,**k:'valid'})()
        return manifest,responses,tokenizer

    def test_empty_or_missing_evidence_never_passes(self):
        manifest,responses,tokenizer=self.fixture()
        for field,value in [('output_ids',[]),('prompt_token_ids',[999])]:
            rows=copy.deepcopy(responses);rows[0][field]=value
            with self.assertRaises(AssertionError):analysis.quality_wave(manifest,rows,'B',0,tokenizer)
        rows=copy.deepcopy(responses);rows[1]['meta_info']['id']=rows[0]['meta_info']['id']
        with self.assertRaises(AssertionError):analysis.quality_wave(manifest,rows,'B',0,tokenizer)
        with self.assertRaises(AssertionError):analysis.quality_wave(manifest,responses[:-1],'B',0,tokenizer)

    def test_ordered_by_request_id_and_first_difference(self):
        manifest,responses,tokenizer=self.fixture()
        result=analysis.quality_wave(manifest,list(reversed(responses)),'B',0,tokenizer)
        self.assertEqual(result,[[7]*128]*16)
        changed=copy.deepcopy(result);changed[3][5]=8
        self.assertEqual(analysis.differences(result,changed),[
            dict(case=3,common_prefix_tokens=5,left_token=7,right_token=8)])


if __name__=='__main__':unittest.main()
