"""Build the small direct-HIP-allocation IPC helper separately from production."""
import hashlib,json,subprocess,sys,os
from pathlib import Path
os.environ['ROCM_HOME']='/opt/rocm'
os.environ['ROCM_PATH']='/opt/rocm'
import torch
from torch.utils.cpp_extension import load
root=Path(__file__).resolve().parent
source=root/'ipc.cu';tag=hashlib.sha256(source.read_bytes()).hexdigest()[:12]
directory=root/('ipc-build-optrocm-'+tag);directory.mkdir(exist_ok=True)
module=load(name='dsv4_h16_ipc_'+tag,sources=[str(source)],build_directory=str(directory),
    extra_cflags=['-O2'],extra_cuda_cflags=['-O2','--offload-arch=gfx90a'],verbose=True)
record=dict(module=module.__file__,name=module.__name__,source=str(source),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    module_sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
(root/'ipc-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
