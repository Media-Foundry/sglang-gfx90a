"""Preserve evidence for a component winner pending model/service validation."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
root=Path(__file__).resolve().parent;target=root/'cooperative-analysis.json';assert not target.exists()
def get(name):
    data=json.loads((root/name).read_text());assert data['status']=='complete',name
    return data
validation=get('cooperative-validation.json');boundary=get('boundary.json')
assert len(validation['checkpoint_fn'])==86 and len(validation['shapes'])==14
for case in validation['shapes']:
    for c in case['checks']:
        assert all(c[k] for k in ['permutation_exact','prefix_exact','replay_exact','changed_input_exact'])
def finish_ast(name):
    module=ast.parse((root/name).read_text())
    return ast.dump(next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='finish'))
assert finish_ast('boundary.py')==finish_ast('verify_cooperative.py')
rows=[]
for name in ['cooperative-screen.json','padded-screen.json','cooperative-split-screen.json']:
    for case in get(name)['cases']:
        for c in case['candidates']:
            a,b=c['median_ms']['A'],c['median_ms']['B']
            rows.append(dict(source=name,m=case['m'],name=c['name'],a_ms=a,b_ms=b,latency_reduction_pct=100*(1-b/a)))
bound=[]
for c in boundary['cases']:
    a,b=c['median_ms']['A'],c['median_ms']['B']
    bound.append(dict(m=c['m'],split=c['split'],a_ms=a,b_ms=b,latency_reduction_pct=100*(1-b/a),scratch_bytes=c['scratch_bytes']))
tools=Path('/opt/rocm/llvm/bin')
for tool,args,name in [('llvm-readelf',['--notes'],'cooperative.metadata.txt'),('llvm-objdump',['-d','--mcpu=gfx90a'],'cooperative.disassembly.txt')]:
    content=subprocess.check_output([str(tools/tool),*args,str(root/'cooperative.hsaco')],text=True)
    (root/name).write_text(content)
asm=(root/'cooperative.disassembly.txt').read_text()
assert 'v_mfma_f32_16x16x4f32' in asm and 'global_load_dwordx4' in asm
archive=root/'cooperative-compiler-evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tf:
    for name in ['cooperative.metadata.txt','cooperative.disassembly.txt']:tf.add(root/name,arcname=name)
checks=[c for row in validation['checkpoint_fn'] for c in row['checks']]
result=dict(status='component-pass-not-integrated',production_input_tps_unchanged=8964.910269,
    component_rows=rows,boundaries=bound,checkpoint_fn_count=86,shape_count=14,
    checkpoint_fn_error={str(s):dict(max_abs=max(c['max_abs'] for c in checks if c['split']==s),
        max_relative_l2=max(c['relative_l2'] for c in checks if c['split']==s)) for s in [4,16]},
    limitations=['FP32 summation order differs from production','Random activations with real Fn are not live model inputs',
                 'No whole-model logits, semantic quality, capacity or E2E acceptance yet'],
    pci_bus=validation['pci_bus'],files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'cooperative.hsaco',archive,root/'cooperative-initial.tar.gz',root/'padded-initial.tar.gz']})
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
