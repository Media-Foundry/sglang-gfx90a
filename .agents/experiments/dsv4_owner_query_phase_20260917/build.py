"""Compile isolated explicit-Lt C ABI; freeze source, library and command."""
import hashlib
import json
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parent
build=root/'build-v1';build.mkdir(exist_ok=False)
source=root/'direct_lt.cpp';output=build/'libdsv4_owner_lt.so'
command=['/opt/rocm/bin/hipcc','-O3','-std=c++17','--offload-arch=gfx90a','-fPIC','-shared',
    str(source),'-I/opt/rocm/include','-L/opt/rocm/lib','-Wl,-rpath,/opt/rocm/lib','-lhipblaslt','-o',str(output)]
with (build/'build.log').open('x') as log:
    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
library=Path('/opt/rocm/lib/libhipblaslt.so').resolve(strict=True)
record=dict(command=command,source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    output=str(output),output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    hipblaslt=str(library),hipblaslt_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
    direct_index=3952,rocblas_index=3953,
    dependencies=subprocess.check_output(['ldd',str(output)],text=True))
(build/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
