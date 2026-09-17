"""Freeze projection/ABI evidence; large historical inputs and binaries stay local."""
import hashlib
import json
from pathlib import Path
import tarfile
root=Path(__file__).resolve().parent;repo=root.parents[2]
for name in ('oracle','confirm','abi'):
    r=json.loads((root/(name+'.json')).read_text())
    assert r['status']=='complete'
    assert hashlib.sha256((root/(name+'.py')).read_bytes()).hexdigest()==r['source_sha256']
assert json.loads((root/'confirm.json').read_text())['exact_cases']==240
assert json.loads((root/'abi.json').read_text())['graph_mutations']==100
files=[p for p in root.iterdir() if p.is_file() and p.suffix in ('.py','.cpp','.json','.log')]
files += [root/'planner-count-failure.tar.gz',root/'abi-stream-guard-evidence.tar.gz']
files += [root/'build-v1/manifest.json',root/'build-v1/build.log']
files += [repo/'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_owner.py',
          repo/'python/sglang/kernels/ops/attention/dsv4/gfx90a_indexer_producer.py',
          repo/'python/sglang/kernels/ops/attention/dsv4/gfx90a_rocblas_linear.py',
          root.parent/'dsv4_qkv_fixed_solution_20260917/oracle.py']
capture=root.parent/'dsv4_c16_owner_producer_20260915/capture/oracle'
files.append(capture/'report.json')
record=json.loads((capture/'report.json').read_text())
assert hashlib.sha256((capture/'inputs.pt').read_bytes()).hexdigest()==record['inputs_file_sha256']
archive=root/'evidence.tar.gz';entries=[]
with tarfile.open(archive,'x:gz') as out:
    for path in sorted(set(files)):
        name=str(path.relative_to(repo));out.add(path,arcname=name)
        entries.append(dict(path=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
manifest=dict(archive=archive.name,bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=entries,
    local_input=dict(path=str(capture/'inputs.pt'),sha256=record['inputs_file_sha256']),
    build='build.py rebuilds the small shim; compiled shared object excluded',
    production_changed=False,service_measurement=False)
(root/'archive-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))
