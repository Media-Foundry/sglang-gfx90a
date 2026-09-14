"""Archive raw evidence and the exact experimental implementation, without deletion."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import psutil

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]


def main():
    archive=ROOT/'measurement-evidence.tar.gz'
    assert not archive.exists()
    for mode in ('atomic','fixed','atomic-matched'):
        assert (ROOT/mode/'complete.json').exists()
        state=json.loads(next((ROOT/mode).glob('*.state.json')).read_text())
        try:
            proc=psutil.Process(state['pid'])
            assert proc.create_time()!=state['birth'] or proc.status()==psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
    gpu=json.loads(subprocess.check_output(['amd-smi','process','--json']))
    assert not any(isinstance(p.get('process_info'),dict)
                   for g in gpu for p in g.get('process_list',[]))
    (ROOT/'final-gpu.json').write_text(json.dumps(gpu,indent=2)+'\n')
    sources=[
        'python/sglang/kernels/ops/moe/gfx90a_bf16_batched_moe.py',
        'python/sglang/kernels/ops/moe/gfx90a_ck_fixed_slot.py',
        'python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_ck_fixed_slot.cuh',
    ]
    contract={p:dict(bytes=(REPO/p).stat().st_size,
        sha256=hashlib.sha256((REPO/p).read_bytes()).hexdigest()) for p in sources}
    (ROOT/'source-contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    subprocess.run([sys.executable,str(ROOT/'analyze.py')],check=True)
    summary=json.loads((ROOT/'summary.json').read_text())
    entries={name:(ROOT/name,entry) for name,entry in summary['artifact_manifest'].items()}
    for name,entry in contract.items(): entries['source/'+name]=(REPO/name,entry)
    p=ROOT/'summary.json'
    entries[p.name]=(p,dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for name,(path,entry) in entries.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
            tar.add(path,arcname=name,recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        assert len(tar.getmembers())==len(entries)
        for name,(_,entry) in entries.items():
            data=tar.extractfile(name).read()
            assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256']
    index=dict(archive=archive.name,bytes=archive.stat().st_size,
               sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),members=len(entries))
    (ROOT/'evidence-index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(index)


if __name__=='__main__':main()
