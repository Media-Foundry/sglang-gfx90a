"""K32 integrated owner eight-rank oracle, not a service throughput result.

Exercises production admission, packing, scoring, Top-K, RCCL, reconstruction
and its full score/ID diagnostic on independently renumbered physical pages.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import statistics
import traceback

os.environ.setdefault('SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER', '3')
os.environ['SGLANG_DSV4_C4_PREFILL_OWNER_K32']='1'
import torch
import torch.distributed as dist
import triton
import triton.language as tl
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_owner import host_plan, forward
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_runtime_m import reuse_runtime_m
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_query_reuse import prefill_query_reuse4
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512
from sglang.kernels.ops.kvcache.triton_store_cache import triton_fused_store_indexer
from sglang.srt.layers.attention.dsv4 import indexer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('gloo', timeout=datetime.timedelta(minutes=10))
    assert dist.get_world_size() == 8
    group = dist.new_group(backend='nccl', timeout=datetime.timedelta(minutes=10))
    repo = Path(__file__).resolve().parents[3]
    paths = [Path(__file__), *(repo / p for p in (
        'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_owner.py',
        'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_owner_k32.py',
        'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_runtime_m.py',
        'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_query_reuse.py',
        'python/sglang/kernels/ops/attention/dsv4/topk.py',
        'python/sglang/kernels/jit/csrc/deepseek_v4/topk_deterministic_hip.cuh'))]
    sources = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = dict(scope=__doc__, status='running', sources=sources, cases=[])
    def save():
        if rank == 0:
            args.output.write_text(json.dumps(result, indent=2)+'\n')
    shuffle = indexer.INDEXER_K_CACHE_PRESHUFFLE_TILE if indexer.aiter_can_use_preshuffle_paged_mqa() else 0
    fp16, fnuz = indexer.envs.SGLANG_DSV4_GFX90A_INDEXER_FP16_DOT.get(), indexer.is_fp8_fnuz()
    result['contract'] = dict(shuffle=shuffle, fp16=fp16, fnuz=fnuz)
    save()
    try:
        for name, width, ext, prefix in [
            ('8k-control', 2048, [8192]*4, [0]*4),
            ('16k', 4096, [16384, 16384], [0, 0]),
            ('32k-ragged', 8192, [32767], [1]),
            ('mixed-prefix', 8192, [8191, 8193, 8190, 8194], [0, 4096, 16384, 24574]),
        ]:
            torch.manual_seed(20260915)
            m, np = sum(ext), width//64
            cpu_lens, ids, valid, inv = host_plan(ext, prefix, rank)
            rowids = torch.from_numpy(ids).cuda()
            valid = torch.from_numpy(valid).cuda()
            inv = torch.from_numpy(inv).cuda()
            lengths = torch.from_numpy(cpu_lens).cuda()
            owner = torch.repeat_interleave(torch.arange(len(ext)), torch.tensor(ext)).cuda()
            storage = torch.empty((m, np+3), device='cuda', dtype=torch.int32)
            pages = storage[:, :np]
            pages.copy_(owner[:, None]*np+torch.arange(np, device='cuda'))
            q = torch.randn(m, 1, 64, 128, device='cuda').to(indexer.FP8_DTYPE)
            w = torch.randn(m, 64, device='cuda')
            cache = torch.empty(np*len(ext), 8448, device='cuda', dtype=torch.uint8)
            values = torch.randn(len(cache)*64, 128, device='cuda')
            locations = torch.arange(len(cache)*64, device='cuda', dtype=torch.int32)
            triton_fused_store_indexer(values, cache, locations, 64)
            # Check replicated synthetic inputs before rank-local page renumbering.
            digest = hashlib.sha256()
            for t in (q.view(torch.uint8), w, lengths, pages, cache):
                digest.update(t.cpu().contiguous().numpy().tobytes())
            hashes = [None]*8
            dist.all_gather_object(hashes, digest.hexdigest())
            assert len(set(hashes)) == 1, hashes
            perm = torch.arange(len(cache), device='cuda').roll(17*rank)
            inverse = torch.empty_like(perm); inverse[perm] = torch.arange(len(perm), device='cuda')
            cache = cache.index_select(0, perm)
            pages.copy_(inverse[pages.long()])
            cap = len(ids)
            assert 0 < cap < 8192
            kw = dict(block_s=16, preshuffle_tile=shuffle, dot_fp16=fp16,
                      fp8_fnuz=fnuz, query_group_size=16, runtime_m=True, allow_wide=True)
            full_raw = torch.empty((m, 512), device='cuda', dtype=torch.int32)
            full_phys = torch.empty_like(full_raw)
            local_raw = torch.empty((cap, 512), device='cuda', dtype=torch.int32)
            local_phys = torch.empty_like(local_raw)
            gathered = torch.empty((cap*8, 512), device='cuda', dtype=torch.int32)
            raw, phys = torch.empty_like(full_raw), torch.empty_like(full_raw)

            def baseline():
                scores = prefill_query_reuse4(q, cache.view(-1,64,1,132), w, lengths, pages, width, **kw)
                assert scores is not None
                topk_transform_512(scores, lengths, pages, full_phys, 64, full_raw)
                return scores

            batch = SimpleNamespace(extend_seq_lens_cpu=ext, extend_prefix_lens_cpu=prefix)
            metadata = SimpleNamespace()
            def candidate():
                ok = forward(q=q, cache=cache.view(-1,64,1,132), weights=w,
                    lengths=lengths, pages=pages, width=width, output=phys, raw_output=raw,
                    batch=batch, metadata=metadata, rank=rank, group=group,
                    preshuffle_tile=shuffle, dot_fp16=fp16, fp8_fnuz=fnuz)
                assert ok, 'Production owner rejected the fixture'
                assert getattr(metadata,'_gfx90a_owner_k32_logged',False), 'K32 selector did not execute'

            os.environ['SGLANG_DSV4_C4_PREFILL_QUERY_OWNER_WIDE']='0'
            if width > 2048:
                assert not forward(q=q, cache=cache.view(-1,64,1,132), weights=w,
                    lengths=lengths, pages=pages, width=width, output=phys, raw_output=raw,
                    batch=batch, metadata=metadata, rank=rank, group=group,
                    preshuffle_tile=shuffle, dot_fp16=fp16, fp8_fnuz=fnuz)
            os.environ['SGLANG_DSV4_C4_PREFILL_QUERY_OWNER_WIDE']='1'
            os.environ['SGLANG_DSV4_DEBUG_PREFILL_OWNER_CHECK']='1'
            checks=[]
            for mutation in range(3):
                if mutation == 1:
                    w.mul_(torch.linspace(.7,1.3,64,device='cuda'))
                    q.copy_(q.view(torch.uint8).roll(1,dims=2).view(q.dtype))
                if mutation == 2: q.zero_()  # exact cutoff ties
                a=baseline();candidate()
                assert bool(torch.isfinite(a).all())
                assert torch.equal(full_raw,raw) and torch.equal(full_phys,phys)
                ref=full_raw.clone();dist.broadcast(ref,src=0,group=group)
                assert torch.equal(full_raw,ref), 'logical IDs differ across TP'
                checks.append(dict(mutation=mutation,scores_byte_exact=True,logical_physical_exact=True))
                del a,ref
            os.environ['SGLANG_DSV4_DEBUG_PREFILL_OWNER_CHECK']='0'
            # Non-tie inputs for timing; ensure final correctness on this input too.
            q.copy_(torch.randn(q.shape,device='cuda').to(q.dtype))
            baseline();candidate()
            assert torch.equal(full_raw,raw) and torch.equal(full_phys,phys)
            samples={'A':[],'B':[]}
            for _ in range(3):
                for arm,fn in [('A',baseline),('B',candidate),('B',candidate),('A',baseline)]:
                    dist.barrier()
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(3): fn()
                    end.record();end.synchronize()
                    samples[arm].append(begin.elapsed_time(end)/3)
            ranks=[None]*8;dist.all_gather_object(ranks,samples)
            rankmax={arm:[max(r[arm][i] for r in ranks) for i in range(6)] for arm in samples}
            centers={arm:statistics.median(v) for arm,v in rankmax.items()}
            result['cases'].append(dict(name=name,rows=m,width=width,local_rows=cap,
                inputs_sha256=hashes,checks=checks,per_rank_ms=ranks,rankmax_ms=rankmax,
                median_ms=centers,speedup=centers['A']/centers['B'],exchange_bytes=gathered.numel()*4,
                baseline_scratch_bytes=m*width*4,owner_scratch_bytes=cap*width*4))
            save()
            if rank==0:print(name,centers,'speedup',centers['A']/centers['B'],flush=True)
        assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in sources.items())
        dist.barrier()
        result['status']='complete';save()
    except BaseException:
        result['status']='failed';result['error']=traceback.format_exc();save();raise
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
