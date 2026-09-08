#!/usr/bin/env python3
"""CPU checks of actual eligibility and moved stream fork; no GPU import."""
import ast
from pathlib import Path

root=Path(__file__).resolve().parents[2]
path=root/'python/sglang/srt/distributed/device_communicators/dsv4_ar_experiment.py'
tree=ast.parse(path.read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)
           and n.name in ('eligible','shared_after_topk_eligible')]
ns={}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),ns)
f=ns['shared_after_topk_eligible']
base=dict(enabled=True,hip=True,arch='gfx90a',decode=True,batch_size=32,
          native=True,tp_size=8,ep_size=1,dsv4=True)
assert f(**base)
for key,values in dict(enabled=[False],hip=[False],arch=['gfx942','gfx950'],
    decode=[False],batch_size=[0,1,2,4,8,16,24,64],native=[False],
    tp_size=[1,2,4],ep_size=[2,4,8],dsv4=[False]).items():
    for value in values:
        assert not f(**(base|{key:value})),(key,value)
model=ast.parse((root/'python/sglang/srt/models/deepseek_v2.py').read_text())
method=next(n for n in ast.walk(model) if isinstance(n,ast.FunctionDef)
            and n.name=='forward_normal_dual_stream')
forks=[n for n in method.body if isinstance(n,ast.If)
       and ast.unparse(n.test) in ('not shared_after_topk','shared_after_topk')]
assert len(forks)==2
for flag in (False,True):
    class Stream:
        def __init__(self): self.waits=[]
        def wait_stream(self,current): self.waits.append(current)
    alt=Stream()
    obj=type('MoE',(),{'alt_stream':alt,'layer_id':20})()
    for block in forks:
        exec(compile(ast.Module(body=[block],type_ignores=[]),'fork','exec'),
             dict(shared_after_topk=flag,self=obj,current_stream='main'))
    assert alt.waits==['main']
source=ast.unparse(method)
assert source.index('if not shared_after_topk:') < source.index('router_logits =')
assert source.index('mark(18)') < source.index('if shared_after_topk:') < source.index('deferred_finalize =')
print('PASS: native TP8/M32-only guard; exactly one fork; delayed fork after TopK')
