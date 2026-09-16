"""Record inspected JIT object and decoded AMDGPU instructions (no GPU work)."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'isa.json';assert not target.exists()
build=Path('/home/pc/.cache/sglang/jit/gfx90a/sgl_kernel_jit_gfx90a_ck_fixed_slot/build-2005df2485acc01c/deps-f921a1c8054b407e')
dependencies=json.loads((build/'sgl_deps.json').read_text())
source=repo/'python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_ck_fixed_slot.cuh'
digest=hashlib.sha256(source.read_bytes()).hexdigest()
assert ['kernels','csrc/deepseek_v4/gfx90a_ck_fixed_slot.cuh',digest] in dependencies
module=build/'sgl_kernel_jit_gfx90a_ck_fixed_slot.so'
device=root/'isa/device.o'
disasm=subprocess.check_output(['/opt/rocm/llvm/bin/llvm-objdump','-d','--demangle',str(device)],text=True)
notes=subprocess.check_output(['/opt/rocm/llvm/bin/llvm-readelf','-n',str(device)],text=True)
blocks={};current=None
for line in disasm.splitlines():
    if line and not line[0].isspace() and line.endswith('>:'):
        current=line
        blocks[current]=[]
    elif current:blocks[current].append(line)
selected={key:'\n'.join(lines) for key,lines in blocks.items() if 'ck_slot_reduce_vec4_kernel' in key}
assert len(selected)==2
for code in selected.values():assert code.count('global_load_dwordx4')==6
assert any('global_store_dwordx2' in code for key,code in selected.items() if '__hip_bfloat16' in key)
assert any('global_store_dwordx4' in code for key,code in selected.items() if '<float>' in key)
result=dict(source=str(source),source_sha256=digest,module=str(module),module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
    extracted_device_sha256=hashlib.sha256(device.read_bytes()).hexdigest(),
    selected_disassembly=selected,amdgpu_notes=notes,
    scope='Inspected JIT source dependency matches measured source; not hardware utilization counters')
target.write_text(json.dumps(result,indent=2)+'\n');print('ISA inspection recorded',target)
