#!/usr/bin/env python3
"""Full raw-FP4 expansion + variable-M CK MoE at production shapes."""

import argparse
import os
import statistics

import torch

from sglang.kernels.ops.moe.gfx90a_bf16_batched_moe import (
    _ck_weight_workspaces,
    _jit_dequant,
    gfx90a_bf16_ck_moe,
)
from aiter.ops.shuffle import shuffle_weight

E,T,H,I=256,6,4096,512


def time_ms(fn, n=3):
    for _ in range(2): fn()
    torch.cuda.synchronize();a=torch.cuda.Event(True);b=torch.cuda.Event(True)
    a.record()
    for _ in range(n):fn()
    b.record();b.synchronize();return a.elapsed_time(b)/n


def routes(kind, device, m):
    if kind == "balanced":
        ids=torch.arange(m*T,device=device,dtype=torch.int32).reshape(m,T)%E
    else:
        # A deliberately difficult but valid distribution: three hot experts
        # plus three diverse experts per token.
        hot=torch.tensor([0,1,2],device=device,dtype=torch.int32).expand(m,3)
        tail=(torch.arange(m*3,device=device,dtype=torch.int32).reshape(m,3)%253)+3
        ids=torch.cat((hot,tail),dim=1)
    return ids,torch.rand((m,T),device=device,dtype=torch.float32)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--m",type=int,default=8192)
    parser.add_argument("--blocks",type=int,nargs="+",default=[832,1248,1664])
    parser.add_argument("--reference-first-token",action="store_true")
    parser.add_argument("--allow-drift",action="store_true")
    parser.add_argument("--small-scales",action="store_true")
    parser.add_argument("--layout-probe",action="store_true")
    parser.add_argument("--verify-direct-shuffle-only",action="store_true")
    args=parser.parse_args();m=args.m
    d=torch.device("cuda");torch.manual_seed(20260902)
    x=torch.randn((m,H),device=d,dtype=torch.bfloat16)
    w13=torch.randint(0,256,(E,2*I,H//2),device=d,dtype=torch.uint8)
    s13=torch.randint(118,132,(E,2*I,H//32),device=d,dtype=torch.uint8)
    w2=torch.randint(0,256,(E,H,I//2),device=d,dtype=torch.uint8)
    s2=torch.randint(118,132,(E,H,I//32),device=d,dtype=torch.uint8)
    if args.verify_direct_shuffle_only:
        raw=torch.empty((E,2*I,H),dtype=torch.bfloat16,device=d)
        direct=torch.empty_like(raw)
        mod=_jit_dequant(E,2*I,H,args.blocks[0])
        mod.run(w13,s13.reshape(E,2*I,H//32),raw)
        expected=shuffle_weight(raw,layout=(16,16))
        mod.run_shuffled(w13,s13.reshape(E,2*I,H//32),direct)
        torch.cuda.synchronize()
        delta=(expected.float()-direct.float()).abs()
        print(
            f"DIRECT_SHUFFLE exact={torch.equal(expected,direct)} "
            f"max_abs={delta.max().item():.7g} mean_abs={delta.mean().item():.7g}"
        )
        return
    if args.small_scales:
        s13.fill_(112)
        s2.fill_(112)
    if args.layout_probe:
        weight13=torch.zeros((E,2*I,H),dtype=torch.bfloat16,device=d)
        weight2=torch.zeros((E,H,I),dtype=torch.bfloat16,device=d)
        gate_values=torch.linspace(0.25,8.0,I,dtype=torch.float32,device=d).bfloat16()
        weight13[:,:I,0]=gate_values
        weight13[:,I:,0]=1
        x.zero_();x[:,0]=1
        _ck_weight_workspaces[torch.cuda.current_device()]=(weight13,weight2)
    for kind in ("balanced","skewed"):
        ids,tw=routes(kind,d,m);out=torch.empty((m,H),device=d,dtype=torch.bfloat16)
        for blocks in args.blocks:
            fn=lambda:gfx90a_bf16_ck_moe(x,ids,tw,w13,s13,w2,s2,out=out,blocks=blocks)
            fn();torch.cuda.synchronize();witness=out.clone();fn();torch.cuda.synchronize()
            if not torch.equal(witness,out):
                replay_delta=(witness.float()-out.float()).abs()
                print(
                    f"kind={kind} replay_max_abs={replay_delta.max().item():.7g} "
                    f"replay_mean_abs={replay_delta.mean().item():.7g}"
                )
                if not args.allow_drift:raise RuntimeError(f"{kind}: replay unstable")
            samples=[time_ms(fn) for _ in range(5)]
            print(f"m={m} kind={kind} blocks={blocks} median_ms={statistics.median(samples):.3f} samples={samples}")
            if args.reference_first_token and blocks == args.blocks[0]:
                import aiter.fused_moe as aiter_fused_moe
                weight13, weight2 = _ck_weight_workspaces[torch.cuda.current_device()]
                if (
                    os.getenv("AITER_DSV4_DEBUG_SHUFFLE_BF16_WEIGHTS", "0") == "1"
                    and os.getenv("AITER_DSV4_DEBUG_KEEP_BF16_WEIGHTS", "0") != "1"
                ):
                    weight13=torch.empty((E,2*I,H),dtype=torch.bfloat16,device=d)
                    weight2=torch.empty((E,H,I),dtype=torch.bfloat16,device=d)
                    _jit_dequant(E,2*I,H,blocks).run(
                        w13,s13.reshape(E,2*I,H//32),weight13
                    )
                    _jit_dequant(E,H,I,blocks).run(
                        w2,s2.reshape(E,H,I//32),weight2
                    )
                internal = gfx90a_bf16_ck_moe(
                    x, ids, tw, w13, s13, w2, s2, out=None, blocks=blocks
                )
                torch.cuda.synchronize()
                reference = torch.zeros((H,), dtype=torch.float32, device=d)
                stage1 = aiter_fused_moe._dsv4_debug_last_stage1
                if stage1 is not None:
                    gu0 = torch.mv(weight13[int(ids[0, 0].item())], x[0]).float()
                    gate0 = gu0[:I].clamp(max=10.0)
                    up0 = gu0[I:].clamp(-10.0, 10.0)
                    act0 = (torch.nn.functional.silu(gate0) * up0).bfloat16()
                    stage1_delta = (stage1[0, 0].float() - act0.float()).abs()
                    sorted_delta = (
                        torch.sort(stage1[0, 0].float()).values
                        - torch.sort(act0.float()).values
                    ).abs()
                    print(
                        f"kind={kind} stage1_ref_max_abs={stage1_delta.max().item():.7g} "
                        f"stage1_ref_mean_abs={stage1_delta.mean().item():.7g} "
                        f"stage1_candidate_absmax={stage1[0,0].float().abs().max().item():.7g} "
                        f"stage1_reference_absmax={act0.float().abs().max().item():.7g} "
                        f"stage1_sorted_max_abs={sorted_delta.max().item():.7g} "
                        f"stage1_sorted_mean_abs={sorted_delta.mean().item():.7g}"
                    )
                    matches=[]
                    for slot_probe in range(T):
                        gu_probe=torch.mv(
                            weight13[int(ids[0,slot_probe].item())],x[0]
                        ).float()
                        g_probe=gu_probe[:I].clamp(max=10.0)
                        u_probe=gu_probe[I:].clamp(-10.0,10.0)
                        ref_probe=(torch.nn.functional.silu(g_probe)*u_probe).float()
                        swapped_probe=(
                            torch.nn.functional.silu(u_probe.clamp(max=10.0))
                            * g_probe.clamp(-10.0,10.0)
                        ).float()
                        matches.append((
                            slot_probe,
                            torch.nn.functional.cosine_similarity(
                                stage1[0,0].float(),ref_probe,dim=0
                            ).item(),
                            torch.nn.functional.cosine_similarity(
                                stage1[0,0].float(),swapped_probe,dim=0
                            ).item(),
                        ))
                    print(f"kind={kind} stage1_slot_matches={matches}")
                    cand_order=torch.argsort(stage1[0,0].float())
                    ref_order=torch.argsort(act0.float())
                    corrected0=torch.empty_like(stage1[0,0])
                    corrected0[ref_order]=stage1[0,0][cand_order]
                    corrected0_delta=(corrected0.float()-act0.float()).abs()
                    expert1=int(ids[1,0].item())
                    gu1=torch.mv(weight13[expert1],x[1]).float()
                    act1=(
                        torch.nn.functional.silu(gu1[:I].clamp(max=10.0))
                        * gu1[I:].clamp(-10.0,10.0)
                    ).bfloat16()
                    corrected1=torch.empty_like(stage1[1,0])
                    corrected1[ref_order]=stage1[1,0][cand_order]
                    corrected1_delta=(corrected1.float()-act1.float()).abs()
                    print(
                        f"kind={kind} inferred_unpermute_t0_max={corrected0_delta.max().item():.7g} "
                        f"t0_mean={corrected0_delta.mean().item():.7g} "
                        f"t1_max={corrected1_delta.max().item():.7g} "
                        f"t1_mean={corrected1_delta.mean().item():.7g}"
                    )
                    if args.layout_probe:
                        distances=(
                            stage1[0,0].float()[:,None]-act0.float()[None,:]
                        ).abs()
                        permutation=distances.argmin(dim=1)
                        unique=int(torch.unique(permutation).numel())
                        nearest=distances.gather(1,permutation[:,None]).squeeze(1)
                        torch.save(permutation.cpu(),"/tmp/dsv4_ck_stage1_perm.pt")
                        print(
                            f"LAYOUT_PROBE unique={unique}/{I} "
                            f"nearest_max={nearest.max().item():.7g} "
                            f"nearest_mean={nearest.mean().item():.7g} "
                            f"first64={permutation[:64].cpu().tolist()}"
                        )
                for slot in range(T):
                    expert = int(ids[0, slot].item())
                    gu = torch.mv(weight13[expert], x[0]).float()
                    gate = gu[:I].clamp(max=10.0)
                    up = gu[I:].clamp(-10.0, 10.0)
                    activated = (torch.nn.functional.silu(gate) * up).bfloat16()
                    down = torch.mv(weight2[expert], activated).float()
                    reference.add_(down, alpha=float(tw[0, slot].item()))
                reference = reference.bfloat16()
                delta = (out[0].float() - reference.float()).abs()
                final_sorted_delta=(
                    torch.sort(out[0].float()).values
                    - torch.sort(reference.float()).values
                ).abs()
                print(
                    f"kind={kind} first_token_max_abs={delta.max().item():.6g} "
                    f"mean_abs={delta.mean().item():.6g} "
                    f"cos={torch.nn.functional.cosine_similarity(out[0].float(), reference.float(), dim=0).item():.9f} "
                    f"candidate_absmax={out[0].float().abs().max().item():.6g} "
                    f"reference_absmax={reference.float().abs().max().item():.6g} "
                    f"candidate_nnz={(out[0] != 0).sum().item()} "
                    f"sorted_max_abs={final_sorted_delta.max().item():.7g} "
                    f"sorted_mean_abs={final_sorted_delta.mean().item():.7g}"
                )
                internal_delta = (internal[0].float() - reference.float()).abs()
                print(
                    f"kind={kind} internal_max_abs={internal_delta.max().item():.6g} "
                    f"mean_abs={internal_delta.mean().item():.6g} "
                    f"cos={torch.nn.functional.cosine_similarity(internal[0].float(), reference.float(), dim=0).item():.9f} "
                    f"internal_absmax={internal[0].float().abs().max().item():.6g} "
                    f"internal_nnz={(internal[0] != 0).sum().item()}"
                )


if __name__=="__main__":main()
