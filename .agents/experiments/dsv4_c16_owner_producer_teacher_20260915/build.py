"""Same real-code tasks, shorter code tails, fixed saved control continuation."""
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

root=Path(__file__).resolve().parent
source=root.parent/'dsv4_c16_owner_producer_service_20260915/A-quality'
old=json.loads((source/'inputs.json').read_text())
answers=sorted(json.loads((source/'quality-0.json').read_text()),key=lambda x:int(x['meta_info']['id'].rsplit('-',1)[-1]))
tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
def encode(prompt):
    return tokenizer.encode('<｜begin▁of▁sentence｜><｜User｜>'+prompt+'<｜Assistant｜></think>',add_special_tokens=False)
requests=[]
for req,answer in zip(old['requests'],answers,strict=True):
    assert encode(req['prompt'])==req['input_ids']==answer['prompt_token_ids']
    continuation=answer['output_ids'];assert len(continuation)==256
    body,sep,tail=req['prompt'].rpartition('\n```');assert sep and not tail
    lines=body.splitlines();removed=[]
    prompt=req['prompt'];ids=encode(prompt)
    while len(ids)>7936:
        removed.append(lines.pop())
        prompt='\n'.join(lines)+'\n```'
        ids=encode(prompt)
    assert 7800<=len(ids)<=7936
    combined=ids+continuation
    assert max(combined)<129279 and len(combined)<=8192 and ids[0]==0 and ids.count(0)==1
    requests.append(dict(index=req['index'],task=req['task'],prompt=prompt,
        base_input_ids=ids,continuation_ids=continuation,input_ids=combined,
        logprob_start_len=len(ids)-1,removed_code_tail_lines=list(reversed(removed)),
        source_candidates=req['source_candidates'],
        input_sha256=hashlib.sha256(json.dumps(combined,separators=(',',':')).encode()).hexdigest()))
result=dict(format='fixed-control-continuation-v1',
    caveat='Continuation comes from the unshortened prompt control; this is numerical comparison, not held-out perplexity or accuracy.',
    source_manifest_sha256=hashlib.sha256((source/'inputs.json').read_bytes()).hexdigest(),
    source_answers_sha256=hashlib.sha256((source/'quality-0.json').read_bytes()).hexdigest(),requests=requests)
out=root/'inputs.json';assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n')
print('combined lengths',[len(r['input_ids']) for r in requests], 'total',sum(len(r['input_ids']) for r in requests))
