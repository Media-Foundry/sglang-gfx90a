"""CPU-only scope/report contracts for the asynchronous prefill marker probe."""
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from sglang.kernels.ops.debug.dsv4_prefill_markers import DIR,active,eligible,instrument,report


class MarkerTests(unittest.TestCase):
    def test_default_is_identity(self):
        def fn():return 1
        with patch.dict(os.environ,{DIR:''}):self.assertIs(instrument(fn),fn)
        self.assertFalse(active())

    def test_scope(self):
        cfg=NS(model_type='deepseek_v4',num_hidden_layers=43,hidden_size=4096)
        ps=NS(tp_size=8,attn_tp_size=8,moe_ep_size=1,attn_cp_size=1)
        runner=NS(model_config=NS(hf_text_config=cfg),ps=ps)
        mode=NS(is_extend_without_speculative=lambda:True)
        batch=NS(spec_algorithm=None,forward_mode=mode,input_ids=NS(shape=(32768,)))
        self.assertTrue(eligible(runner,batch))
        for obj,name,value in [(cfg,'model_type','deepseek_v41'),(cfg,'num_hidden_layers',61),
                               (ps,'tp_size',4),(ps,'moe_ep_size',2),(ps,'attn_cp_size',2)]:
            old=getattr(obj,name);setattr(obj,name,value)
            self.assertFalse(eligible(runner,batch));setattr(obj,name,old)
        batch.spec_algorithm=NS(is_none=lambda:False)
        self.assertFalse(eligible(runner,batch));batch.spec_algorithm=None
        batch.input_ids.shape=(128,);self.assertFalse(eligible(runner,batch))
        batch.input_ids.shape=(32768,);mode.is_extend_without_speculative=lambda:False
        self.assertFalse(eligible(runner,batch))

    def test_report_keeps_raw_and_marks_missing_stages(self):
        ticks=[[1000+layer*100+i for i in range(8)]+[0]*24 for layer in range(43)]
        ticks.append([999,6000]+[0]*30)
        value=report(ticks,0.20004,dict(rank=0,sequence=1,wall_clock_khz=25000))
        self.assertAlmostEqual(value['us_per_tick'],0.04)
        self.assertTrue(all(r['coarse_valid'] for r in value['layers']))
        ticks[20][4]=0
        value=report(ticks,0.20004,dict(wall_clock_khz=25000))
        self.assertFalse(value['layers'][20]['coarse_valid'])
        self.assertIsNone(value['layers'][20]['coarse_us'])
        ticks[43][1]=0
        with self.assertRaises(AssertionError):report(ticks,0.20004,dict(wall_clock_khz=25000))


if __name__=='__main__':unittest.main()
