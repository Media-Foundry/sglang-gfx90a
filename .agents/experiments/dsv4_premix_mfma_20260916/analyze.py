"""Archive compiler evidence and summarize the bounded MFMA rejection."""
import hashlib
import json
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
target=root/'analysis.json'
assert not target.exists()
obj=Path('/home/pc/.cache/sglang/jit/gfx90a/sgl_kernel_jit_dsv4_premix_mfma_split_screen/build-0657f02f1dc536ad/deps-d96cf8b467a0ffb1/cuda_0.o')
tools=Path('/opt/rocm/llvm/bin')
def run(*args):return subprocess.check_output([str(x) for x in args],text=True)
metadata=run(tools/'llvm-readelf','--notes',root/'split.hsaco')
disassembly=run(tools/'llvm-objdump','-d','--mcpu=gfx90a',root/'split.hsaco')
(root/'split.metadata.txt').write_text(metadata)
(root/'split.disassembly.txt').write_text(disassembly)
initial=json.loads((root/'screen.json').read_text())
split=json.loads((root/'split-screen.json').read_text())
assert initial['status']==split['status']=='complete'
rows=[]
for kind,data in [('unsplit',initial),('split',split)]:
    for case in data['cases']:
        for result in case['candidates']:
            assert result['permutation_exact'] and result['replay100_exact']
            a,b=result['median_ms']['A'],result['median_ms']['B']
            rows.append(dict(kind=kind,m=case['m'],name=result['name'],reference_ms=a,candidate_ms=b,
                slowdown=b/a,max_abs=max(c['max_abs'] for c in result['checks']),
                max_relative_l2=max(c['relative_l2'] for c in result['checks']),
                production_bit_exact=all(c['bit_exact'] for c in result['checks'])))
assert 'v_mfma_f32_16x16x4f32' in disassembly
result=dict(decision='Reject scalar-loaded FP32 MFMA and split-K variants for production large-M pre-mix',
    scope='Does not reject all MFMA designs: cooperative/vectorized supply is untested',
    actual_pci_bus=split['pci_bus'],hip_visible_devices=split['hip_visible_devices'],
    gpu_label_correction='Initial mapping/screen physical_gcd=5 denotes HIP_VISIBLE_DEVICES ordinal, not rocm-smi card index. Measured HIP device PCI B3:00.0 maps to rocm-smi card7.',
    rows=rows,production_speed_unchanged_input_tps=8964.910269,
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [obj,root/'split.hsaco',root/'initial-screen.tar.gz',root/'split.metadata.txt',root/'split.disassembly.txt']})
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
