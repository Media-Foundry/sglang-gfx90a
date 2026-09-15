"""Separate clean one-instance CK module; never modify installed AIter binaries."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
aiter=Path('/home/pc/pytorch/third_party/aiter')
ninja=aiter/'aiter/jit/build/module_moe_ck2stages_b16_b16_preshuffle_off_f32_silu_no_mulWeightStage2_/build/build.ninja'
common=aiter/'csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.cuh'
original=aiter/'3rdparty/composable_kernel/include/ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp'
overlay=root/'overlay/ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp'
needle='IsInputGemm ? InMemoryDataOperationEnum::Set : InMemoryDataOperationEnum::AtomicAdd;'
assert overlay.read_text()==original.read_text().replace(needle,'InMemoryDataOperationEnum::Set; // EXPERIMENT: unique assignment slots only.')
paths=[Path(__file__),root/'entry.cu',overlay,original,common,ninja]
scatter_path='ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v7r3_scatter.hpp'
paths += [root/'overlay'/scatter_path, aiter/'3rdparty/composable_kernel/include'/scatter_path]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
tag=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:12]
name='dsv4_ck_unique_'+tag
build=root/('build-'+tag);build.mkdir(exist_ok=False)
lines=ninja.read_text().splitlines()
flags=shlex.split(next(l.split(' = ',1)[1] for l in lines if l.startswith('cuda_cflags = ')))
flags=[f'-DTORCH_EXTENSION_NAME={name}' if f.startswith('-DTORCH_EXTENSION_NAME=') else
       '--offload-arch=gfx90a' if f.startswith('--offload-arch=') else f for f in flags]
flags=[f for f in flags if not f.startswith('-DMOE_STAGE2_ASM_DIR=')]
flags.insert(0,'-I'+str(root/'overlay'))
link=shlex.split(next(l.split(' = ',1)[1] for l in lines if l.startswith('ldflags = ')))
obj=build/'entry.o';so=build/(name+'.so')
commands=[['/opt/rocm/bin/hipcc',*flags,'-c',str(root/'entry.cu'),'-o',str(obj)],
          ['/opt/rocm/bin/hipcc',str(obj),*link,'-o',str(so)]]
record=dict(name=name,sources=hashes,commands=commands,module=str(so),status='building')
manifest=build/'manifest.json'
manifest.write_text(json.dumps(record,indent=2)+'\n')
for i,cmd in enumerate(commands):
    with (build/f'command-{i}.log').open('w') as log:
        proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if proc.returncode:
        record.update(status='failed',failed_command=i,returncode=proc.returncode)
        manifest.write_text(json.dumps(record,indent=2)+'\n')
        raise RuntimeError(f'build failed: {build}/command-{i}.log')
record.update(status='complete',module_sha256=hashlib.sha256(so.read_bytes()).hexdigest())
manifest.write_text(json.dumps(record,indent=2)+'\n')
print(manifest,flush=True)
