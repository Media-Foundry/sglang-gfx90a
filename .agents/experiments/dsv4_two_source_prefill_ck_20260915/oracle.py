"""Isolated checked ABI. Not imported by any server selector."""
from functools import cache
from pathlib import Path
import torch
from torch.utils.cpp_extension import include_paths
from sglang.kernels.jit.utils import load_jit

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]

@cache
def module():
    return load_jit('gfx90a_prefill_two_source_ck_oracle_v1',
        cuda_files=['debug/gfx90a_prefill_two_source_ck_oracle.cuh'],
        cuda_wrappers=[('run', 'sglang::prefill_two_source_oracle::Entry::run')],
        extra_cuda_cflags=['-O3','-std=c++20','-DCK_ENABLE_BF16','-DCK_USE_XDL',
                          '-DSGLANG_DSV4_CK_SENTINEL_ORACLE=1'],
        extra_include_paths=[str(ROOT),*include_paths(),
            '/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/include',
            '/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/library/include'])

class Runner:
    def __init__(self,q,pkv,pi,pp,ekv,ei,ep,sink):
        assert 'gfx90a' in torch.cuda.get_device_properties(q.device).gcnArchName
        self.inputs = (q,pkv,pi,pp,ekv,ei,ep,sink)
        self.out = torch.empty_like(q, memory_format=torch.contiguous_format)
        self.scratch = torch.empty(q.shape[0]*2*8*514*4, device=q.device, dtype=torch.uint8)
        self.combined = torch.empty(max(1,pi.numel()+ei.numel()), device=q.device, dtype=torch.int32)
        self.ptr = torch.empty(q.shape[0]+1, device=q.device, dtype=torch.int32)
        self.mod = module()
    def __call__(self, splits=2):
        self.mod.run(*self.inputs,self.out,self.scratch,self.combined,self.ptr,512**-.5,splits)
        return self.out

def reference(q,pkv,pi,pp,ekv,ei,ep,sink):
    result=[]
    pptr=pp.cpu().tolist();eptr=ep.cpu().tolist()
    for row in range(q.shape[0]):
        p=pi[pptr[row]:pptr[row+1]];e=ei[eptr[row]:eptr[row+1]]
        p=p[(p>=0)&(p<pkv.shape[0])];e=e[(e>=0)&(e<ekv.shape[0])]
        values=torch.cat((pkv[p.long()],ekv[e.long()]),dim=0).float()
        scores=q[row].float()@values.T*(512**-.5)
        prob=torch.softmax(torch.cat((scores,sink[:,None]),dim=1),dim=1)[:,:-1]
        result.append(prob@values)
    return torch.stack(result)

def check_mapping(runner):
    q,pkv,pi,pp,ekv,ei,ep,sink=runner.inputs
    pi,pp,ei,ep=[t.cpu().tolist() for t in (pi,pp,ei,ep)]
    expected=[]
    for row in range(q.shape[0]):
        expected.extend(s if 0<=s<pkv.shape[0] else -1 for s in pi[pp[row]:pp[row+1]])
        expected.extend((s|(1<<30)) if 0<=s<ekv.shape[0] else -1 for s in ei[ep[row]:ep[row+1]])
    assert runner.ptr.cpu().tolist()==[a+b for a,b in zip(pp,ep,strict=True)]
    assert runner.combined[:len(expected)].cpu().tolist()==expected
