"""CPU-only inspection of the unique dependency-matching runtime JIT artifact."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

root=Path(__file__).resolve().parent;repo=root.parents[2]
source=repo/'python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_fp4_bf16_direct_rows.cuh'
source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
cache=Path('/home/pc/.cache/sglang/jit/gfx90a/sgl_kernel_jit_gfx90a_bf16_direct_rows')
candidates=[]
for metadata in cache.glob('build-*/deps-*/sgl_deps.json'):
    if ['kernels','csrc/deepseek_v4/gfx90a_fp4_bf16_direct_rows.cuh',source_hash] in json.loads(metadata.read_text()):
        candidates.append(metadata.parent)
assert len(candidates)==1,candidates
build=candidates[0];out=root/'compiler';out.mkdir(exist_ok=False)
module=build/'sgl_kernel_jit_gfx90a_bf16_direct_rows.so';llvm=Path('/opt/rocm/llvm/bin')
commands=[
    [str(llvm/'llvm-objcopy'),'--dump-section',f'.hip_fatbin={out}/fatbin',str(module),'/dev/null'],
    [str(llvm/'clang-offload-bundler'),'-unbundle','-type=o','-targets=hipv4-amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-',
     f'-input={out}/fatbin',f'-output={out}/device.o']]
for command in commands:subprocess.run(command,check=True)
for name,command in [('assembly',[str(llvm/'llvm-objdump'),'-d','--demangle',str(out/'device.o')]),
                     ('notes',[str(llvm/'llvm-readelf'),'-n',str(out/'device.o')])]:
    (out/(name+'.txt')).write_text(subprocess.check_output(command,text=True));commands.append(command)
assembly=(out/'assembly.txt').read_text()
counts={mnemonic:len(re.findall(r'\b'+mnemonic+r'\b',assembly)) for mnemonic in
        ('global_load_dwordx4','global_store_dwordx4','s_barrier','ds_write_b128','ds_read_b128')}
record=dict(module=str(module),module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
    source=str(source),source_sha256=source_hash,commands=commands,instruction_occurrences=counts,
    scope='Dependency-matched local runtime compilation; static instruction counts, not runtime counters',
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()})
(out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(counts),flush=True)
