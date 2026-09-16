"""Decode the tested HIP object; no GPU work and no installed files modified."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent;repo=root.parents[2];out=root/'hip-compiler';out.mkdir(exist_ok=False)
build=Path('/home/pc/.cache/sglang/jit/gfx90a/sgl_kernel_jit_gfx90a_mhc_post_wave/build-1c50ae0a68b7bebe/deps-f10be676b2476397')
source=repo/'python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_mhc_post_wave.cuh'
source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
assert ['kernels','csrc/deepseek_v4/gfx90a_mhc_post_wave.cuh',source_hash] in json.loads((build/'sgl_deps.json').read_text())
module=build/'sgl_kernel_jit_gfx90a_mhc_post_wave.so';llvm=Path('/opt/rocm/llvm/bin')
commands=[
    [str(llvm/'llvm-objcopy'),'--dump-section',f'.hip_fatbin={out}/fatbin',str(module),'/dev/null'],
    [str(llvm/'clang-offload-bundler'),'-unbundle','-type=o','-targets=hipv4-amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-',f'-input={out}/fatbin',f'-output={out}/device.o'],
]
for cmd in commands:subprocess.run(cmd,check=True)
for name,cmd in [('assembly',[str(llvm/'llvm-objdump'),'-d','--demangle',str(out/'device.o')]),
                 ('notes',[str(llvm/'llvm-readelf'),'-n',str(out/'device.o')])]:
    (out/(name+'.txt')).write_text(subprocess.check_output(cmd,text=True));commands.append(cmd)
record=dict(module=str(module),module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
    source=str(source),source_sha256=source_hash,commands=commands,
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
(out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
