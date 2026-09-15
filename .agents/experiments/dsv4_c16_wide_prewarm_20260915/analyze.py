"""Recompute cross-process priming identities; not a service-throughput test."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
prime,warm,cold=[json.loads((root/'v2'/f'{name}.json').read_text()) for name in ('prime','warm','cold')]
utility=json.loads((root/'utility.json').read_text())
assert prime['source_sha256']==warm['source_sha256']==cold['source_sha256']
assert prime['dtype_contract']==warm['dtype_contract']==cold['dtype_contract']
assert len(prime['records'])==len(warm['records'])==len(cold['records'])==len(utility['records'])==3
rows=[]
for p,w,c,u in zip(prime['records'],warm['records'],cold['records'],utility['records'],strict=True):
    assert (p['width'],p['pages'],p['page_stride'])==(w['width'],w['pages'],w['page_stride'])==(c['width'],c['pages'],c['page_stride'])
    assert p['hash']==w['hash']==c['hash']==u['artifact']
    assert p['hsaco_sha256']==w['hsaco_sha256']==c['hsaco_sha256']==u['hsaco_sha256']
    assert set(w['input_hashes'])=={'q','cache','weights','lengths','pages'}
    assert w['input_hashes']==c['input_hashes'] and w['output_sha256']==c['output_sha256']
    assert p['kernel_launches']==p['tensor_allocated_delta']==0
    rows.append(dict(width=p['width'],page_stride=p['page_stride'],m=w['m'],
                     cold_first_s=c['first_use_s'],primed_first_s=w['first_use_s'],
                     prime_compile_s=p['compile_s'],same_inputs_outputs_binary=True))
result=dict(rows=rows,scope=__doc__,service_cold_ttft_verified=False)
out=root/'analysis.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
