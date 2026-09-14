"""Freeze CK stage-2 inputs, isolate atomic drift, and test unique-slot output.

Diagnostic only: fixed-slot scratch is M*6*4096*4 bytes. No production selector.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m', type=int, default=8192)
    p.add_argument('--replays', type=int, default=20)
    p.add_argument('--kind', choices=('balanced','skewed','random'), default='balanced')
    p.add_argument('--fused', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    assert os.environ.get('HIP_VISIBLE_DEVICES') == '4'
    # This isolated test requires no pre-existing GPU users.
    gpu = json.loads(subprocess.check_output(['amd-smi','process','--json']))
    assert not any(isinstance(x.get('process_info'),dict)
                   for g in gpu for x in g.get('process_list',[]))
    assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx90a')
    os.environ.update(SGLANG_DSV4_GFX90A_BF16_CK_STAGE2_FP32='1',
        SGLANG_DSV4_GFX90A_BF16_CK_BLOCK64_V1='1',
        AITER_DSV4_DEBUG_SHUFFLE_BF16_WEIGHTS='1',
        SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT='0')
    from sglang.kernels.ops.moe.gfx90a_bf16_batched_moe import gfx90a_bf16_ck_moe
    module = importlib.import_module(
        'aiter.jit.module_moe_ck2stages_b16_b16_preshuffle_off_f32_silu_no_mulWeightStage2_')
    original = module.ck_moe_stage2
    captured = []
    def capture(*a, **kw):
        assert not kw
        captured[:] = a
        return original(*a)
    module.ck_moe_stage2 = capture
    m,h,e,t,i = args.m,4096,256,6,256
    torch.manual_seed(20260914)
    x = torch.randn((m,h),device='cuda',dtype=torch.bfloat16)
    w13 = torch.randint(0,256,(e,2*i,h//2),device='cuda',dtype=torch.uint8)
    w2 = torch.randint(0,256,(e,h,i//2),device='cuda',dtype=torch.uint8)
    # Nonconstant scales: constant-scale fixtures hid an earlier layout bug.
    s13 = torch.randint(118,121,(e,2*i,h//32),device='cuda',dtype=torch.uint8)
    s2 = torch.randint(118,121,(e,h,i//32),device='cuda',dtype=torch.uint8)
    ids = torch.arange(m*t,device='cuda',dtype=torch.int32).reshape(m,t)%e
    if args.kind == 'skewed':
        ids = torch.cat((torch.tensor([0,1,2],device='cuda',dtype=torch.int32).expand(m,3),
                         3+torch.arange(m*3,device='cuda',dtype=torch.int32).reshape(m,3)%253),dim=1)
    elif args.kind == 'random':
        ids = torch.rand((m,e),device='cuda').topk(t,dim=1).indices.int()
    weights = torch.softmax(torch.randn((m,t),device='cuda'),dim=-1)
    y = gfx90a_bf16_ck_moe(x,ids,weights,w13,s13,w2,s2)
    torch.cuda.synchronize()
    module.ck_moe_stage2 = original
    a = list(captured)
    assert len(a) in (16,17) and a[7] == t, (len(a),a[7])
    a[0] = a[0].clone()  # Freeze stage-1 independently of the helper workspace.
    a[3] = a[3].clone()
    a[4] = a[4].clone()
    a[5] = a[5].clone()
    a[12] = a[12].clone() if a[12] is not None else None
    a[6] = torch.zeros((m,h),device='cuda',dtype=torch.float32)
    results = dict(m=m, kind=args.kind, fused=args.fused, bitwise_checked=True,
        replays=args.replays, gpu_before=gpu,
        head=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),
        module=module.__file__, module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        valid_ids=a[5].tolist(), stage1_nonzero=int(torch.count_nonzero(a[0]).item()))
    # Fixed logical inputs, but rerun the sorter and stage-1 too. Compare only
    # initialized sorter prefixes, never undefined padding outside valid_ids.
    valid=int(a[5][0])
    full=[]
    first_output=y.clone()
    module.ck_moe_stage2=capture
    try:
        for _ in range(5):
            current=gfx90a_bf16_ck_moe(x,ids,weights,w13,s13,w2,s2)
            now=list(captured)
            full.append(dict(stage1_exact=torch.equal(now[0].view(torch.int16),a[0].view(torch.int16)),
                sorted_ids_exact=torch.equal(now[3][:valid],a[3][:valid]),
                sorted_weights_exact=torch.equal(now[12][:valid],a[12][:valid]),
                sorted_experts_exact=torch.equal(now[4][:valid//64],a[4][:valid//64]),
                valid_exact=torch.equal(now[5],a[5]),
                output_exact=torch.equal(current.view(torch.int16),first_output.view(torch.int16)),
                output_changed=int(torch.count_nonzero(current!=first_output).item())))
    finally:
        module.ck_moe_stage2=original
    results['full_stage_replays']=full

    def baseline():
        a[6].zero_()
        original(*a)
        return a[6]

    def compare(fn):
        ref = fn().clone()
        torch.cuda.synchronize()
        records = []
        for _ in range(args.replays):
            out = fn()
            delta=(out-ref).abs()
            records.append(dict(exact=torch.equal(out.view(torch.int32),ref.view(torch.int32)),
                changed=int(torch.count_nonzero(delta).item()), max_abs=float(delta.max()),
                bf16_changed=int(torch.count_nonzero(out.bfloat16()!=ref.bfloat16()).item())))
        return ref, records

    base, results['atomic_replays'] = compare(baseline)
    # Packed token ID: low24=token, high8=original TopK slot. Change both the
    # input row view and token address: one virtual token per original slot.
    # Stage-2 still multiplies the same sorted router weights; KBatch stays 1.
    token = a[3] & 0xffffff
    slot = a[3] >> 24
    b = list(a)
    b[0] = a[0].view(m*t,1,i)
    b[3] = torch.where(token<m, token*t+slot, m*t).to(torch.int32)
    b[6] = torch.zeros((m*t,h),device='cuda',dtype=torch.float32)
    b[7] = 1
    reduced = torch.empty((m,h),device='cuda',dtype=torch.float32)
    if args.fused:
        from sglang.kernels.ops.moe.gfx90a_ck_fixed_slot import fixed_slot_module
        mod=fixed_slot_module()
        expected=b[3].clone()
        mod.remap(a[3],a[5],b[3],m)
        assert torch.equal(b[3][:valid],expected[:valid])
    def fixed():
        b[6].zero_()
        original(*b)
        partial = b[6].view(m,t,h)
        if args.fused:
            mod.reduce_float(partial,reduced)
        else:
            reduced.copy_(partial[:,0])
            for k in range(1,t): reduced.add_(partial[:,k])
        return reduced
    fix,results['fixed_slot_replays'] = compare(fixed)
    difference=(fix-base).abs()
    results['fixed_vs_atomic'] = dict(max_abs=float(difference.max()),
        relative_l2=float(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(base)),
        bf16_changed=int(torch.count_nonzero(fix.bfloat16()!=base.bfloat16()).item()),
        finite=bool(torch.isfinite(fix).all()))
    # Selected-row sequential FP32 reference for the frozen BF16 intermediate.
    # Undo (16,16) B preshuffle via the exact inverse of AIter shuffle_weight.
    from aiter.ops.shuffle import shuffle_weight
    # Shuffle vectorization depends on dtype item size: use BF16 labels in three
    # limbs so the address map is exact for the actual weight dtype.
    linear = torch.arange(h*i,device='cuda',dtype=torch.int64).reshape(1,h,i)
    # Three base128 limbs are exactly representable in BF16.
    permutation = sum(shuffle_weight(((linear//(128**k))%128).bfloat16(),layout=(16,16)).flatten().long()*(128**k) for k in range(3))
    assert torch.equal(torch.sort(permutation).values,linear.flatten())
    inverse=torch.argsort(permutation)
    errors=[]
    for row in (0,1,m//2,m-1):
        ref=torch.zeros(h,device='cuda',dtype=torch.float32)
        for k in range(t):
            expert=int(ids[row,k])
            logical_w=a[2][expert].flatten()[inverse].view(h,i)
            ref.add_(torch.mv(logical_w.float(),a[0][row,k].float())*weights[row,k])
        errors.append(dict(row=row,max_abs=float((fix[row]-ref).abs().max()),
            relative_l2=float(torch.linalg.vector_norm(fix[row]-ref)/torch.linalg.vector_norm(ref))))
    results['fp32_selected_row_reference']=errors
    for name,fn in [('atomic',baseline),('fixed_slot',fixed)]:
        timings=[]
        for _ in range(5):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):fn()
            end.record();end.synchronize();timings.append(start.elapsed_time(end)/5)
        results[name+'_ms']=timings
    results['fixed_scratch_bytes']=b[6].numel()*4
    if args.fused:
        os.environ['SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT']='1'
        integrated=gfx90a_bf16_ck_moe(x,ids,weights,w13,s13,w2,s2)
        results['integrated_helper_exact']=torch.equal(integrated.view(torch.int16),fix.bfloat16().view(torch.int16))
        assert results['integrated_helper_exact']
        os.environ['SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT']='0'
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): fixed()
        reference=fixed().clone()
        exact=0
        for _ in range(100):
            graph.replay()
            exact+=int(torch.equal(reduced.view(torch.int32),reference.view(torch.int32)))
        results['graph_replays_exact']=exact
        mutations=[]
        for _ in range(5):
            a[0].add_(0.015625)
            reference=fixed().clone()
            graph.replay()
            mutations.append(torch.equal(reduced.view(torch.int32),reference.view(torch.int32)))
        results['graph_input_mutation_exact']=mutations
        os.environ['SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT']='1'
        permutation_rows=torch.randperm(m,device='cuda')
        inverse_rows=torch.argsort(permutation_rows)
        permuted=gfx90a_bf16_ck_moe(x[permutation_rows],ids[permutation_rows],
                                  weights[permutation_rows],w13,s13,w2,s2)[inverse_rows]
        results['fixed_full_stage_row_permutation']=dict(
            exact=torch.equal(permuted.view(torch.int16),integrated.view(torch.int16)),
            changed=int(torch.count_nonzero(permuted!=integrated).item()),
            max_abs=float((permuted.float()-integrated.float()).abs().max()))
        if m < 36864:
            extended=gfx90a_bf16_ck_moe(torch.cat((x,x[:1])),torch.cat((ids,ids[:1])),
                                      torch.cat((weights,weights[:1])),w13,s13,w2,s2)[:m]
            results['fixed_full_stage_append_one_row']=dict(
                exact=torch.equal(extended.view(torch.int16),integrated.view(torch.int16)),
                changed=int(torch.count_nonzero(extended!=integrated).item()),
                max_abs=float((extended.float()-integrated.float()).abs().max()))
        os.environ['SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT']='0'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)


if __name__ == '__main__': main()
