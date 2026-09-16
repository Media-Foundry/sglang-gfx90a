"""Default-off original-V4 TP8 large-prefill route-major CK stages.

Uses verified independent stage1 and unique-Set stage2 binaries. No AIter
global stage override, retained intermediate, or whole-model weight cache.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
import hashlib
import importlib.util
import json
import logging
from pathlib import Path

import torch
from sglang.kernels.jit.utils import cache_once, load_jit

STAGE1 = "moe_ck2stages_gemm1_256x64x64x128_1x4_TypeCast_v1_Nswizzle0_Quant0_MulRoutedWeight0_dsv4silu_B16_B16_B16"
_reference = ContextVar("dsv4_route_producer_reference", default=False)


def in_reference():
    return _reference.get()


@contextmanager
def reference_scope():
    token = _reference.set(True)
    try:
        yield
    finally:
        _reference.reset(token)


def eligible(m, i, *, native_scope, reference, shuffle, keep, raw,
             stage2_fp32, block_m, stage1_kernel, stage2_kernel,
             dsv4_activation, fixed_slot, unique_manifest, route_manifest,
             capturing, has_probe):
    return (native_scope and not reference and not capturing and not has_probe
            and 16384 <= m <= 36864 and i == 256 and shuffle and not keep and not raw
            and stage2_fp32 and block_m == 64 and stage1_kernel == STAGE1
            and not stage2_kernel and dsv4_activation and fixed_slot
            and bool(unique_manifest) and bool(route_manifest))


@lru_cache(maxsize=1)
def load_route(manifest_path):
    record = json.loads(Path(manifest_path).read_text())
    if record['status'] != 'complete':
        raise ValueError('route producer requires a completed independent build')
    if '-DDSV4_ROUTE_MAJOR_STAGE1=1' not in record['commands'][0]:
        raise ValueError('route producer requires route-output stage1, not token-output')
    for name, digest in record['sources'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'route producer source changed since build: {name}')
    path = Path(record['module'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['module_sha256']:
        raise ValueError('route producer binary hash mismatch')
    spec = importlib.util.spec_from_file_location(record['name'], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache_once
def helper():
    return load_jit('gfx90a_ck_route_producer',
        cuda_files=['deepseek_v4/gfx90a_ck_route_producer.cuh'],
        cuda_wrappers=[('metadata','sglang::RouteMetadata::run'),
                       ('reduce','sglang::RouteMajor::bf16')],
        extra_cuda_cflags=['-O3','-fno-fast-math','-ffp-contract=off'])


@cache_once
def announce():
    from sglang.srt.distributed import get_tp_group
    logging.getLogger(__name__).warning(
        'DSV4 route-producer CK selected: rank=%s block_m=64 unique_set=1 fixed_top6=1',
        get_tp_group().rank_in_group)


def forward(hidden, ids, weights, w13, w2, *, route_manifest, unique_manifest, out=None):
    import aiter.fused_moe as fused
    from sglang.kernels.ops.debug.dsv4_ck_unique_store import load_verified

    m, h = hidden.shape
    if not (16384 <= m <= 36864 and h == 4096 and ids.shape == (m, 6)
            and weights.shape == (m, 6) and hidden.dtype == torch.bfloat16
            and w13.shape == (256, 512, 4096) and w2.shape == (256, 4096, 256)
            and w13.dtype == w2.dtype == torch.bfloat16):
        raise ValueError('route producer requires tested TP8 large-prefill BF16 shapes')
    if not torch.version.hip or 'gfx90a' not in torch.cuda.get_device_properties(hidden.device).gcnArchName:
        raise ValueError('route producer requires gfx90a')
    stage1 = load_route(route_manifest)
    stage2 = load_verified(unique_manifest)
    maps = helper()
    si, sw, se, nv, out = fused.moe_sorting(ids, weights, 256, 4096, torch.bfloat16,
                                          64, None, None, 0, out)
    capacity = si.numel()
    inter = torch.empty((capacity, 256), device=hidden.device, dtype=torch.bfloat16)
    identity = torch.empty_like(si)
    inverse = torch.empty((m, 6), device=hidden.device, dtype=torch.int32)
    partial = torch.empty((capacity, 4096), device=hidden.device, dtype=torch.float32)
    maps.metadata(si, nv, identity, inverse)
    stage1.stage1(hidden, w13, si, se, nv, inter)
    stage2.stage2(inter.view(capacity, 1, 256), w2, identity, se, nv, sw, partial)
    maps.reduce(partial, inverse, out)
    announce()
    return out
