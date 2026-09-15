"""Check the exported helper itself has no GPU launch or tensor allocation."""
import json
import os
from pathlib import Path
import subprocess

assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
assert not any(isinstance(p.get('process_info'),dict) for g in owners for p in g.get('process_list',[]))
root=Path(__file__).resolve().parent
os.environ['TRITON_CACHE_DIR']=str(root/'v2/primed-cache')
import torch
import triton
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_prewarm import prewarm_query_reuse4

torch.cuda.set_device(0);torch.cuda.current_stream()
launches=[]
triton.knobs.runtime.launch_enter_hook=lambda *a,**k: launches.append(1)
records=[]
for record in json.loads((root/'v2/prime.json').read_text())['records']:
    before=torch.cuda.memory_allocated()
    c=prewarm_query_reuse4(width=record['width'],page_columns=record['pages'],page_stride=record['page_stride'])
    assert not launches and torch.cuda.memory_allocated()==before
    assert c.hash==record['hash']
    records.append(dict(artifact=c.hash,kernel_launches=0,tensor_allocated_delta=0))
out=root/'helper-check.json';assert not out.exists();out.write_text(json.dumps(records,indent=2)+'\n')
print('Exported helper: three exact artifacts, zero launches, zero tensor allocation')
