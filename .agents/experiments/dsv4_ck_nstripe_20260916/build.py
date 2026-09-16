"""Pinned private CK overlay; no installed source or binary modification."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

root=Path(__file__).resolve().parent
base=root.parent/'dsv4_prefill_extras_build_20260916/build-v1/unique/manifest.json'
contract=json.loads(base.read_text())
assert contract['status']=='complete'
for name,digest in contract['sources'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
aiter=Path('/home/pc/pytorch/third_party/aiter')
relative='ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp'
original=aiter/'3rdparty/composable_kernel/include'/relative
raw=original.read_text()
needle='problem.N * problem.K * (IsInputGemm ? 2 : 1)'
assert raw.count(needle)==2
replacement='(IsInputGemm ? problem.N * 2 : 4096) * problem.K'
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),root/'entry.cu',base,original]}
tag=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()[:12]
name='dsv4_ck_nstripe_'+tag
build=root/('build-'+tag);build.mkdir(exist_ok=False)
overlay=build/'overlay'/relative;overlay.parent.mkdir(parents=True)
overlay.write_text(raw.replace(needle,replacement))
sources[str(overlay)]=hashlib.sha256(overlay.read_bytes()).hexdigest()
obj,so,deps=build/'entry.o',build/(name+'.so'),build/'entry.d'
compile_command=list(contract['commands'][0])
compile_command.insert(1,'-I'+str(build/'overlay'))
for i,arg in enumerate(compile_command):
    if arg.startswith('-DTORCH_EXTENSION_NAME='):compile_command[i]='-DTORCH_EXTENSION_NAME='+name
for flag,value in [('-c',root/'entry.cu'),('-o',obj),('-MF',deps)]:
    compile_command[compile_command.index(flag)+1]=str(value)
link_command=list(contract['commands'][1]);link_command[1]=str(obj)
link_command[link_command.index('-o')+1]=str(so)
commands=[compile_command,link_command]
record=dict(status='building',name=name,module=str(so),sources=sources,commands=commands,
            base_manifest=str(base),toolchain=contract['toolchain'])
manifest=build/'manifest.json'
def save():manifest.write_text(json.dumps(record,indent=2)+'\n')
save()
for i,command in enumerate(commands):
    print('BUILD',i,build,flush=True)
    with (build/f'command-{i}.log').open('w') as log:
        status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
    if status:
        record.update(status='failed',returncode=status,failed_command=i);save()
        raise RuntimeError(f'Build failed: {build}/command-{i}.log')
for dependency in shlex.split(deps.read_text().replace('\\\n',' ').split(':',1)[1]):
    path=Path(dependency).resolve();record['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
record.update(status='complete',module_sha256=hashlib.sha256(so.read_bytes()).hexdigest());save()
index=root/'build.json';assert not index.exists()
index.write_text(json.dumps(dict(manifest=str(manifest)),indent=2)+'\n')
print(manifest,flush=True)
