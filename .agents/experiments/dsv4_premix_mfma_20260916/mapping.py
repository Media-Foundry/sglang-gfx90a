"""Small exact-integer MFMA lane/accumulator mapping oracle before real inputs."""
import hashlib
import json
import os
from pathlib import Path
assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
from module import load
root=Path(__file__).resolve().parent;target=root/'mapping.json';assert not target.exists()
torch.manual_seed(2026091602);mod=load()
assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx90a')
report=dict(status='running',physical_gcd=5,cases=[],sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (root/'premix.cuh',root/'module.py',Path(__file__))})
for m,n,k in ((16,16,4),(1,24,4),(7,32,8),(17,24,64),(65,24,128)):
    a=torch.randint(-7,8,(m,k)).to(torch.bfloat16)
    b=torch.randint(-5,6,(n,k)).to(torch.float32)
    reference=(a.double()@b.double().T).float().cuda()
    a=a.cuda();b=b.cuda();out=torch.empty((m,n),device='cuda')
    checks={}
    for name in ('map0','map1','u4','u8'):
        out.fill_(float('nan'));getattr(mod,name)(a,b,out)
        checks[name]=dict(exact=torch.equal(out.view(torch.int32),reference.view(torch.int32)),max_abs=float((out-reference).abs().max()))
    report['cases'].append(dict(shape=[m,n,k],checks=checks))
    target.write_text(json.dumps(report,indent=2)+'\n')
    assert checks['map0']['exact'] and checks['u4']['exact'] and checks['u8']['exact'],report['cases'][-1]
    print(m,n,k,checks,flush=True)
report['status']='complete';target.write_text(json.dumps(report,indent=2)+'\n')
