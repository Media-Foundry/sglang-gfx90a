"""Production wrapper and MHC call-site oracle; isolated GPU 4, no service overlap."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

assert os.environ.get('HIP_VISIBLE_DEVICES') == '4'
assert os.environ.get('SGLANG_DSV4_PREFILL_MIX_REUSE4') == '1'
assert os.environ.get('SGLANG_DSV4_PREFILL_MIX_PAIR_COLUMNS') == '1'
owners = json.loads(subprocess.check_output(['amd-smi', 'process', '--json']))
assert not any(isinstance(p.get('process_info'), dict)
               for g in owners for p in g.get('process_list', [])), owners

import torch
import triton
import sglang.kernels.ops.layernorm.gfx90a_mhc_premix_pair as pair_module
from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_reuse import premix_reuse4
from sglang.kernels.ops.layernorm.mhc import gfx90a_mhc_pre_mix_from_partials_triton
from sglang.srt.layers.dsv4_prefill_experiments import _mix_reuse, _mix_pair

root = Path(__file__).resolve().parent
repo = root.parents[2]
output = root / 'integrated.json'
assert not output.exists()
source = root.parent / 'dsv4_input_identity_20260914/trace-B1'
fn = torch.load(source / 'layer_0_rank_0_hc_ffn_fn.pt', map_location='cuda', weights_only=True).contiguous()
seed = torch.load(source / 'layer_0_rank_0_ffn_mhc_residual.pt', map_location='cuda', weights_only=True)
original = pair_module.premix8_pair
launches = []

class CountLaunch:
    def __getitem__(self, grid):
        kernel = original[grid]
        def launch(*args, **kwargs):
            launches.append({'m': args[4], 'grid': list(grid)})
            return kernel(*args, **kwargs)
        return launch

pair_module.premix8_pair = CountLaunch()
paths = ['python/sglang/kernels/ops/layernorm/' + f for f in (
    'gfx90a_mhc_premix_pair.py', 'gfx90a_mhc_premix_reuse.py',
    'gfx90a_mhc_premix_reuse8.py', 'mhc.py')]
paths += ['python/sglang/srt/layers/dsv4_prefill_experiments.py']
result = {'status': 'running', 'checks': [], 'sources': {
    p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in paths}}
def save(): output.write_text(json.dumps(result, indent=2) + '\n')
def exact(a, b): return torch.equal(a.view(torch.int32), b.view(torch.int32))
torch.manual_seed(20260915)
save()
try:
    for m in (1, 128, 8191, 8192, 32767, 32768, 65536, 65537):
        x = seed.repeat(triton.cdiv(m, len(seed)), 1, 1)[:m].contiguous()
        rms = x.float().square().reshape(m, 64, 256).sum(-1)
        for group in (4, 8):
            before = len(launches)
            a = premix_reuse4(x, fn, rms, 1e-6, group_size=group, pair_columns=False)
            b = premix_reuse4(x, fn, rms, 1e-6, group_size=group, pair_columns=True)
            expected = int(group == 8 and 8192 <= m <= 65536)
            assert len(launches) - before == expected
            assert exact(a, b)
            result['checks'].append(dict(m=m, group=group, bits_exact=True, pair_launches=expected))
        if 8192 <= m <= 65536:
            os.environ['SGLANG_DSV4_PREFILL_MIX_GROUP_SIZE'] = '8'
            def call(enabled):
                reuse_token = _mix_reuse.set(True)
                pair_token = _mix_pair.set(enabled)
                try:
                    return gfx90a_mhc_pre_mix_from_partials_triton(x, fn, rms, 1e-6)
                finally:
                    _mix_pair.reset(pair_token)
                    _mix_reuse.reset(reuse_token)
            for _ in range(10):
                x.normal_(); fn.normal_(std=.01)
                rms.copy_(x.float().square().reshape(m, 64, 256).sum(-1))
                before = len(launches)
                a, b = call(False), call(True)
                assert len(launches) - before == 1
                assert exact(a, b)
                assert not _mix_pair.get() and not _mix_reuse.get()
            result['checks'].append(dict(m=m, caller_mutations=10, bits_exact=True, pair_launches=10))
        save()
        print(json.dumps(result['checks'][-1]), flush=True)
        del x, rms, a, b
        torch.cuda.empty_cache()
    result['status'] = 'complete'
    result['launches'] = launches
    save()
finally:
    pair_module.premix8_pair = original
