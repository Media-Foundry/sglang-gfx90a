#!/usr/bin/env python3
"""Exact/ABBA micro for AIter M32 A4 sorter default versus no-clear."""
import argparse,statistics,torch
from aiter.fused_moe import moe_sorting,moe_sorting_no_clear
from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import reconstruct_topk_from_counts

def default(ids,w):return moe_sorting(ids,w,256,4096,torch.bfloat16,block_size=4)[:4]
def noclear(ids,w):return moe_sorting_no_clear(ids,w,256,block_size=4)
def exact(a,b,label):
 if not torch.equal(a[3],b[3]):raise RuntimeError(label+' num_valid')
 valid=int(a[3][0]);blocks=(valid+3)//4
 for n,x,y in [('ids',a[0][:valid],b[0][:valid]),('weights',a[1][:valid],b[1][:valid]),('experts',a[2][:blocks],b[2][:blocks])]:
  if not torch.equal(x,y):raise RuntimeError(f'{label} {n} mismatch={(x!=y).sum().item()}')
def cap(fn):
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):o=fn()
 return g,o
def time(g,iters):
 for _ in range(30):g.replay()
 torch.cuda.synchronize();a=torch.cuda.Event(True);b=torch.cuda.Event(True);a.record()
 for _ in range(iters):g.replay()
 b.record();b.synchronize();return a.elapsed_time(b)*1000/iters
def trim(v):return statistics.mean(sorted(v)[1:-1])
def main():
 p=argparse.ArgumentParser();p.add_argument('--recorder',default='/tmp/expert_distribution_recorder_1787803355.1855972.pt');p.add_argument('--mutations',type=int,default=100);p.add_argument('--replays',type=int,default=1000);p.add_argument('--rounds',type=int,default=7);p.add_argument('--iters',type=int,default=500);a=p.parse_args()
 payload=torch.load(a.recorder,map_location='cpu',weights_only=False);real=reconstruct_topk_from_counts(payload['logical_count'][37,34]//8).cuda();ids=real.clone();w=torch.rand(32,6,device='cuda')
 for i in range(a.mutations):
  if i:ids.copy_(torch.topk(torch.rand(32,256,device='cuda'),6,dim=1,sorted=False).indices.int())
  else:ids.copy_(real)
  w.uniform_();x=default(ids,w);y=noclear(ids,w);torch.cuda.synchronize();exact(x,y,f'mutation={i}')
 print(f'CORRECTNESS mutations={a.mutations} exact=True')
 ids.copy_(real);w.uniform_();ga,oa=cap(lambda:default(ids,w));gb,ob=cap(lambda:noclear(ids,w))
 for i in range(a.replays):
  ga.replay();gb.replay()
  if i%100==99:torch.cuda.synchronize();exact(oa,ob,f'replay={i+1}')
 print(f'GRAPH replays={a.replays} exact=True')
 v={'A':[],'B':[]}
 for _ in range(a.rounds):
  for k in ('A','B','B','A'):v[k].append(time(ga if k=='A' else gb,a.iters))
 aa,bb=trim(v['A']),trim(v['B']);print(f'RESULT default_us={aa:.3f} no_clear_us={bb:.3f} saving_us={aa-bb:.3f} gain_pct={(aa/bb-1)*100:.3f} A={v["A"]} B={v["B"]}')
if __name__=='__main__':main()
