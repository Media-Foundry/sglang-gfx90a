"""Read-only comparison of completed quality waves, never infer missing output.

Cross-cycle recurrence is diagnostic context, not a controlled A/B comparison.
Only requests with the exact same explicit input IDs are compared.
"""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
experiments=root.parent
current=json.loads((root/'A1/inputs.json').read_text())['requests']
waves=[]
for name in ('dsv4_c16_post_fused4_20260915','dsv4_c16_premix_reuse_20260915',root.name):
    for arm in ('A1','B','A2'):
        directory=experiments/name/arm
        manifest=directory/'inputs.json'
        if not manifest.exists():continue
        requests=json.loads(manifest.read_text())['requests']
        assert [r['input_ids'] for r in requests]==[r['input_ids'] for r in current]
        for rep in range(2):
            path=directory/f'quality-{rep}.json'
            if not path.exists():continue
            responses=json.loads(path.read_text())
            ordered={int(r['meta_info']['id'].split('-')[-1]):r for r in responses}
            assert len(responses)==len(ordered)==16 and set(ordered)==set(range(16))
            ids=[]
            for case in range(16):
                response=ordered[case];meta=response['meta_info']
                assert response['prompt_token_ids']==current[case]['input_ids']
                assert meta['completion_tokens']==128 and meta['cached_tokens']==0
                tokens=response['output_ids'][-128:];assert len(tokens)==128
                ids.append(tokens)
            waves.append(dict(label=f'{name}/{arm}/{rep}',ids=ids))
result=dict(scope=__doc__,waves=[w['label'] for w in waves],variants={})
for case in range(16):
    buckets={}
    for wave in waves:
        tokens=wave['ids'][case]
        digest=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()
        buckets.setdefault(digest,dict(tokens=tokens,waves=[]))['waves'].append(wave['label'])
    if len(buckets)>1:result['variants'][case]=buckets
(root/'prior-quality-variants.json').write_text(json.dumps(result,indent=2)+'\n')
for case,variants in result['variants'].items():
    print('case',case,'variants',len(variants))
    for digest,data in variants.items():print(digest[:12],data['waves'])
