"""Real layer20 TP8 logits/TopK ownership oracle; not a production selector.

Query projection and compressor stay replicated and are outside BOTH paths.
Candidate times GPU packing, selected logits/TopK, RCCL integer all-gather,
and full-row logical/physical reconstruction. CPU planning is offline here.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_query_reuse import prefill_query_reuse4
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


@triton.jit
def reconstruct(gathered, inverse, lengths, pages, raw, physical,
                M: tl.constexpr, PT: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, 512)
    length = tl.load(lengths + row)
    source = tl.load(inverse + row)
    if length > 512:
        logical = tl.load(gathered + source * 512 + k)
    else:
        logical = tl.where(k < length, length - 1 - k, -1)
    page = tl.load(pages + row * PT + tl.maximum(logical, 0) // 64,
                   mask=logical >= 0, other=0)
    tl.store(raw + row * 512 + k, logical)
    tl.store(physical + row * 512 + k,
             tl.where(logical >= 0, page * 64 + logical % 64, -1))


def plan(lengths):
    # Keep original query16 groups, including boundary trivial rows. Round-robin
    # groups distribute increasing causal lengths; no live D2H selector.
    m = len(lengths)
    groups = [list(range(i, min(i + 16, m))) for i in range(0, m, 16)
              if any(x > 512 for x in lengths[i:i + 16])]
    owners = [[] for _ in range(8)]
    for i, group in enumerate(groups):
        owners[i % 8].extend(group + [-1] * (16 - len(group)))
    capacity = max(map(len, owners))
    inverse = [-1] * m
    for rank, rows in enumerate(owners):
        rows.extend([-1] * (capacity - len(rows)))
        for local, row in enumerate(rows):
            if row >= 0:
                assert inverse[row] == -1
                inverse[row] = rank * capacity + local
    assert all(l <= 512 or inverse[i] >= 0 for i, l in enumerate(lengths))
    return owners, inverse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cycles', type=int, default=3)
    parser.add_argument('--iters', type=int, default=5)
    parser.add_argument('--capture', default='capture-v2')
    parser.add_argument('--analysis', default='analysis-v2.json')
    args = parser.parse_args()
    assert not args.output.exists()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', timeout=datetime.timedelta(minutes=8))
    assert dist.get_world_size() == 8
    comm = dist.new_group(backend='nccl', timeout=datetime.timedelta(minutes=8))
    assert args.cycles > 0 and args.iters > 0
    analysis = json.loads((ROOT / args.analysis).read_text())
    assert analysis['all_checked_replicated_values_exact']
    record = json.loads((ROOT / args.capture / 'data/rank-0-layer-20.json').read_text())
    assert record['logical_decoder_version'] == 2
    path = ROOT / args.capture / 'data' / record['full_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['full_sha256']
    fixture = torch.load(path, weights_only=True)
    meta = fixture['metadata']
    m = record['rows']
    qtype = getattr(torch, record['tensors']['q']['dtype'].split('.')[-1])
    qraw = fixture['q'].cuda()
    q = qraw.view(qtype).reshape(m, 1, 64, 128)
    weights = fixture['weights'].view(torch.float32).reshape(m, 64).cuda()
    lengths_cpu = fixture['seq_lens'].reshape(-1).tolist()
    lengths = fixture['seq_lens'].reshape(-1).to(device='cuda', dtype=torch.int32)
    pages = fixture['page_table'].to(device='cuda', dtype=torch.int32)
    cache = fixture['cache'].cuda()
    # Deliberately DIFFERENT physical page numbering on every rank. Logical-ID
    # exchange must remain correct after each rank applies its own page table.
    perm = torch.arange(cache.shape[0], device='cuda').roll(17 * rank)
    inverse_page = torch.empty_like(perm)
    inverse_page[perm] = torch.arange(len(perm), device='cuda')
    cache = cache.index_select(0, perm)
    pages = inverse_page[pages.long()].to(torch.int32)
    width = max(lengths_cpu)
    owners, inverse = plan(lengths_cpu)
    cap = len(owners[rank])
    rowids = torch.tensor([max(i, 0) for i in owners[rank]], device='cuda', dtype=torch.int64)
    valid = torch.tensor([i >= 0 for i in owners[rank]], device='cuda', dtype=torch.int32)
    inv = torch.tensor(inverse, device='cuda', dtype=torch.int32)
    pqraw = torch.empty((cap, 1, 64, 128), device='cuda', dtype=torch.uint8)
    pq = pqraw.view(qtype)
    pw = torch.empty((cap, 64), device='cuda')
    pl = torch.empty(cap, device='cuda', dtype=torch.int32)
    pp = torch.empty((cap, pages.shape[1]), device='cuda', dtype=torch.int32)
    full_phys = torch.empty((m, 512), device='cuda', dtype=torch.int32)
    full_raw = torch.empty_like(full_phys)
    part_phys = torch.empty((cap, 512), device='cuda', dtype=torch.int32)
    part_raw = torch.empty_like(part_phys)
    gathered = torch.empty((8 * cap, 512), device='cuda', dtype=torch.int32)
    rebuilt_raw, rebuilt_phys = torch.empty_like(full_raw), torch.empty_like(full_phys)
    kw = dict(block_s=16, preshuffle_tile=meta['preshuffle_tile'],
              dot_fp16=meta['dot_fp16'], fp8_fnuz=meta['fp8_fnuz'],
              query_group_size=16, runtime_m=True)

    def logits(q_, w_, l_, p_):
        result = prefill_query_reuse4(q_, cache, w_, l_, p_, width, **kw)
        assert result is not None, (q_.shape, cache.shape, kw)
        return result

    def baseline():
        score = logits(q, weights, lengths, pages)
        topk_transform_512(score, lengths, pages, full_phys, 64, full_raw)
        return score

    def pack():
        torch.index_select(qraw, 0, rowids, out=pqraw)
        torch.index_select(weights, 0, rowids, out=pw)
        torch.index_select(lengths, 0, rowids, out=pl)
        pl.mul_(valid)
        torch.index_select(pages, 0, rowids, out=pp)

    def candidate():
        pack()
        score = logits(pq, pw, pl, pp)
        topk_transform_512(score, pl, pp, part_phys, 64, part_raw)
        dist.all_gather_into_tensor(gathered, part_raw, group=comm)
        reconstruct[(m,)](gathered, inv, lengths, pages, rebuilt_raw, rebuilt_phys,
                           m, pages.stride(0), num_warps=4)
        return score

    def rebuild():
        reconstruct[(m,)](gathered, inv, lengths, pages, rebuilt_raw, rebuilt_phys,
                           m, pages.stride(0), num_warps=4)

    result = dict(status='running',scope=__doc__,fixture_sha256=record['full_sha256'],
                  rows=m,width=width,capacity_per_owner=cap,active_rows=sum(l > 512 for l in lengths_cpu),
                  gathered_bytes=gathered.numel()*4,backend='RCCL all_gather_into_tensor',
                  rank_specific_physical_page_permutation=True,query_projection_unchanged=True,
                  sources={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__), REPO/'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_query_reuse.py',
                                     REPO/'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_runtime_m.py',
                                     REPO/'python/sglang/kernels/ops/attention/dsv4/topk.py',
                                     REPO/'python/sglang/kernels/jit/csrc/deepseek_v4/topk_deterministic_hip.cuh']},checks=[])
    if rank == 0:
        snapshots=ROOT/'oracle-sources';snapshots.mkdir(exist_ok=True)
        for source,digest in result['sources'].items():
            destination=snapshots/(digest+Path(source).suffix)
            data=(REPO/source).read_bytes()
            if destination.exists():assert destination.read_bytes()==data
            else:destination.write_bytes(data)
    def save():
        if rank == 0: args.output.write_text(json.dumps(result, indent=2)+'\n')
    save()
    try:
        # Full-rank comparisons, not sampled rows. Mutate weights without
        # changing routing lengths; check finite scores and bitwise ordering.
        original_weights = weights.clone()
        original_q = qraw.clone()
        for mutation in range(3):
            weights.copy_(original_weights)
            qraw.copy_(original_q)
            if mutation:
                weights.mul_(torch.linspace(.7,1.3,64,device='cuda').pow(mutation))
            if mutation == 2:
                qraw.copy_(original_q.roll(1, dims=2))
            a = baseline()
            b = candidate()
            torch.cuda.synchronize()
            torch.testing.assert_close(rebuilt_raw, full_raw, atol=0, rtol=0)
            torch.testing.assert_close(rebuilt_phys, full_phys, atol=0, rtol=0)
            selected = a.index_select(0, rowids)
            mask = valid.bool()
            assert torch.equal(selected[mask].view(torch.int32), b[mask].view(torch.int32))
            assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
            result['checks'].append(dict(mutation=mutation,all_scores_bits_exact=True,
                                         logical_ids_exact=True,physical_ids_exact=True))
            if rank == 0: print('correctness',result['checks'][-1],flush=True)
            del a,b,selected
        weights.copy_(original_weights)
        qraw.copy_(original_q)
        for _ in range(3): baseline();candidate()
        torch.cuda.synchronize();dist.barrier()
        samples={'baseline':[],'candidate':[]}
        for cycle in range(args.cycles):
            for name, fn in [('baseline',baseline),('candidate',candidate),
                             ('candidate',candidate),('baseline',baseline)]:
                dist.barrier()
                begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(args.iters): fn()
                end.record();end.synchronize()
                samples[name].append(begin.elapsed_time(end)/args.iters)
            if rank == 0: print('ABBA cycle',cycle,flush=True)
        ranks=[None]*8;dist.all_gather_object(ranks,samples)
        result['per_rank_ms']=ranks
        result['rank_max_ms']={name:[max(r[name][i] for r in ranks) for i in range(len(samples[name]))]
                               for name in samples}
        result['median_ms']={name:statistics.median(v) for name,v in result['rank_max_ms'].items()}
        result['speedup']=result['median_ms']['baseline']/result['median_ms']['candidate']
        # Diagnostic component timings are separate rank-max measurements;
        # never sum their medians into the measured complete-chain latency.
        a=baseline();b=candidate()
        operations={
            'baseline_logits':lambda:logits(q,weights,lengths,pages),
            'baseline_topk':lambda:topk_transform_512(a,lengths,pages,full_phys,64,full_raw),
            'pack':pack,
            'owner_logits':lambda:logits(pq,pw,pl,pp),
            'owner_topk':lambda:topk_transform_512(b,pl,pp,part_phys,64,part_raw),
            'exchange':lambda:dist.all_gather_into_tensor(gathered,part_raw,group=comm),
            'reconstruct':rebuild}
        component_samples={}
        for name,fn in operations.items():
            fn();torch.cuda.synchronize()
            component_samples[name]=[]
            for _ in range(3):
                dist.barrier()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(args.iters):fn()
                end.record();end.synchronize()
                component_samples[name].append(begin.elapsed_time(end)/args.iters)
        per_rank_components=[None]*8
        dist.all_gather_object(per_rank_components,component_samples)
        result['diagnostic_component_rankmax_ms']={name:[max(r[name][i] for r in per_rank_components) for i in range(3)] for name in operations}
        # Repeated eager calls are the actual prefill execution mode, not a
        # claim of CUDA/HIP Graph support for this experimental communication.
        for _ in range(25):candidate()
        torch.cuda.synchronize()
        torch.testing.assert_close(rebuilt_raw,full_raw,atol=0,rtol=0)
        torch.testing.assert_close(rebuilt_phys,full_phys,atol=0,rtol=0)
        result['repeated_eager_replay_exact']=25
        assert all(hashlib.sha256((REPO/source).read_bytes()).hexdigest()==digest
                   for source,digest in result['sources'].items())
        dist.barrier()  # All ranks passed final replay and source-integrity checks.
        result['status']='complete';save()
        if rank==0:print(json.dumps({k:result[k] for k in ('status','median_ms','speedup','capacity_per_owner','gathered_bytes')}),flush=True)
    except Exception as error:
        result['status']='failed';result['error']=repr(error);save();raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__': main()
