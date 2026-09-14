"""Locate the candidate's case8 wording branch in committed control evidence."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
archive=root.parent/'dsv4_c16_indexer_qgroup16_20260915/evidence.tar.gz'
with tarfile.open(archive) as tar:
    old_inputs=json.load(tar.extractfile('A2/inputs.json'))
    old_responses=json.load(tar.extractfile('A2/quality-1.json'))
inputs=json.loads((root/'B/inputs.json').read_text())
assert old_inputs==inputs
def case(rows,index):
    return next(r for r in rows if r['meta_info']['id'].endswith('-'+str(index)))
old=case(old_responses,8)
current=case(json.loads((root/'B/quality-0.json').read_text()),8)
other=case(json.loads((root/'B/quality-1.json').read_text()),8)
tokens=lambda r:r['output_ids'][-r['meta_info']['completion_tokens']:]
assert len(tokens(old))==len(tokens(current))==128
assert tokens(current)==tokens(old)
prefix=next(i for i,(a,b) in enumerate(zip(tokens(current),tokens(other))) if a!=b)
prompt=inputs['requests'][8]['prompt']
result=dict(case=8,identical_full_input_manifests=True,
    candidate_wave0_equals_historical_control_128_tokens=True,
    candidate_repeat_common_prefix=prefix,
    old_control_archive=str(archive),old_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    old_control_member='A2/quality-1.json',candidate_wave0_text=current['text'],
    candidate_wave1_text=other['text'],
    input_symbols_present={symbol:symbol in prompt for symbol in (
        'get_available_gpu_memory','cpu_group','ReduceOp.MIN','1 << 30')},
    conclusion='Existing output branch reproduced; does not establish the cause of stochastic whole-model drift or whole-model bitwise determinism.')
(root/'case8-review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
