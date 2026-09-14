"""One-GCD synthetic lifecycle/exactness test, NOT a TP8 performance result."""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace as NS

import torch

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
args=p.parse_args();assert not args.output_dir.exists();args.output_dir.mkdir()
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
os.environ['SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR']=str(args.output_dir.resolve())
os.environ['SGLANG_DSV4_GFX90A_REALTIME_TRACE_GRAPH_ONLY']='0'
from sglang.kernels.ops.debug.dsv4_prefill_markers import active,instrument
from sglang.kernels.ops.debug.gfx90a_realtime_marker import gfx90a_realtime_marker

layers=[NS(self_attn=NS(wo_b=NS()),mlp=NS()) for _ in range(43)]
runner=NS(model=NS(model=NS(layers=layers)),
          model_config=NS(hf_text_config=NS(model_type='deepseek_v4',num_hidden_layers=43,hidden_size=4096)),
          ps=NS(tp_size=8,attn_tp_size=8,moe_ep_size=1,attn_cp_size=1,tp_rank=0))
class Fake:
    model_runner=runner
    @instrument
    def forward(self,batch,x):
        for layer in layers:
            for slot in range(8):gfx90a_realtime_marker(layer._gfx90a_realtime_trace,slot)
        return x*2+1

fake=Fake();torch.manual_seed(7)
batch=NS(spec_algorithm=None,forward_mode=NS(is_extend_without_speculative=lambda:True),
         input_ids=torch.zeros(8192,device='cuda',dtype=torch.int32),batch_size=1,
         extend_seq_lens_cpu=[8192],extend_prefix_lens_cpu=[0])
for i in range(2):
    x=torch.randn(8192,32,device='cuda')
    ref=x*2+1;got=fake.forward(batch,x)
    assert torch.equal(ref,got) and not active()
    target=args.output_dir/f'rank-0-frame-{i+1:04d}.json'
    deadline=time.monotonic()+20
    while not target.exists() and time.monotonic()<deadline:time.sleep(.05)
    assert target.exists(),fake._dsv4_prefill_marker_collector.error
    report=json.loads(target.read_text())
    assert all(row['coarse_valid'] for row in report['layers'])
    assert .03 < report['us_per_tick'] < .05
    print(i,'exact output, valid43 layers; HIP-reported us/tick',report['us_per_tick'],flush=True)
batch.forward_mode.is_extend_without_speculative=lambda:False
assert torch.equal(fake.forward(batch,x),ref)
assert fake._dsv4_prefill_marker_collector.sequence==2
print('Non-extend excluded; no extra frame or numeric change',flush=True)
