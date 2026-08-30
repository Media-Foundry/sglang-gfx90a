#!/usr/bin/env python3
"""Real L20 TP4 M32 wo_a FP16 compute oracle; no production wiring."""
import argparse,statistics,torch
from safetensors import safe_open

M,G,D,R,H=32,2,4096,1024,4096
def deq(w,s):return (w.float()*s.float().repeat_interleave(128,0).repeat_interleave(128,1)).bfloat16()
def cap(fn):
 for _ in range(5):fn()
 torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):o=fn()
 return g,o
def time(g,n):
 for _ in range(20):g.replay()
 torch.cuda.synchronize();a=torch.cuda.Event(True);b=torch.cuda.Event(True);a.record()
 for _ in range(n):g.replay()
 b.record();b.synchronize();return a.elapsed_time(b)*1000/n
def trim(v):return statistics.mean(sorted(v)[1:-1])
def metric(a,b):
 d=(a.float()-b.float());return float(d.abs().max()),float(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(a.float()))
def main():
 p=argparse.ArgumentParser();p.add_argument('--dump',default='/tmp/dsv4_ffn_dump.f3ZQ89');p.add_argument('--model',default='/home/pc/models/modelscope/model-00022-of-00048.safetensors');p.add_argument('--mutations',type=int,default=100);p.add_argument('--replays',type=int,default=1000);p.add_argument('--rounds',type=int,default=7);p.add_argument('--iters',type=int,default=300);a=p.parse_args()
 x=torch.cat([torch.load(f'{a.dump}/layer_20_rank_{i}_attn_inverse_rope.pt',map_location='cpu',weights_only=False) for i in (0,1)],1).reshape(M,G,D).cuda().contiguous();base=x.clone();mut=torch.linspace(-1,1,x.numel(),device='cuda').view_as(x).to(torch.bfloat16)
 with safe_open(a.model,framework='pt',device='cpu') as f:
  wa=deq(f.get_tensor('layers.20.attn.wo_a.weight'),f.get_tensor('layers.20.attn.wo_a.scale'))[:G*R].view(G,R,D).cuda().contiguous()
  wb=deq(f.get_tensor('layers.20.attn.wo_b.weight'),f.get_tensor('layers.20.attn.wo_b.scale'))[:,:G*R].cuda().contiguous()
 wah=wa.half().contiguous();xh=x.half().contiguous()
 def waA():return torch.einsum('mgd,grd->mgr',x,wa)
 def waB():return torch.bmm(x.half().transpose(0,1),wah.transpose(1,2)).transpose(0,1).to(torch.bfloat16)
 def waC():return torch.bmm(xh.transpose(0,1),wah.transpose(1,2)).transpose(0,1).to(torch.bfloat16)
 def full(f):
  def z():return torch.mm(f().flatten(1),wb.t())
  return z
 fs={'A_woa':waA,'B_woa':waB,'C_woa':waC,'A_full':full(waA),'B_full':full(waB),'C_full':full(waC)};gs={};out={}
 for k,f in fs.items():gs[k],out[k]=cap(f)
 worst={'woa_B':[0.,0.],'woa_C':[0.,0.],'full_B':[0.,0.],'full_C':[0.,0.]}
 for i in range(a.mutations):
  x.copy_(base).add_(mut,alpha=((i*1543+17)%2047-1023)/32768.);xh.copy_(x)
  for g in gs.values():g.replay()
  torch.cuda.synchronize()
  for label,aa,bb in [('woa_B','A_woa','B_woa'),('woa_C','A_woa','C_woa'),('full_B','A_full','B_full'),('full_C','A_full','C_full')]:
   m=metric(out[aa],out[bb]);worst[label]=[max(worst[label][0],m[0]),max(worst[label][1],m[1])]
 print('CORRECTNESS mutations=',a.mutations,'worst=',worst)
 x.copy_(base);xh.copy_(x)
 for g in gs.values():g.replay()
 torch.cuda.synchronize();snap={k:v.clone() for k,v in out.items()}
 for i in range(a.replays):
  for g in gs.values():g.replay()
  if i%100==99:
   torch.cuda.synchronize()
   for k in out:
    if not torch.equal(out[k],snap[k]):raise RuntimeError(f'replay {i+1} unstable {k}')
 print(f'GRAPH replays={a.replays} stable=True')
 vals={k:[] for k in fs}
 for _ in range(a.rounds):
  for group in [('A_woa','B_woa','C_woa','C_woa','B_woa','A_woa'),('A_full','B_full','C_full','C_full','B_full','A_full')]:
   for k in group:vals[k].append(time(gs[k],a.iters))
 for k in vals:print(f'RESULT profile={k} trimmed_us={trim(vals[k]):.3f} samples={vals[k]}')
 for stage in ('woa','full'):
  aa=trim(vals[f'A_{stage}'])
  for q in ('B','C'):
   z=trim(vals[f'{q}_{stage}']);print(f'DECISION stage={stage} profile={q} gain_pct={(aa/z-1)*100:.3f} passes_32_7={z<32.7}')
if __name__=='__main__':main()
