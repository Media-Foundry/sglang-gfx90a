"""Check user content through the same official chat encoder, not raw encode."""
import hashlib
import json
from pathlib import Path
from build_inputs import AutoTokenizer, load_encoder

root = Path(__file__).resolve().parent
model = Path('/home/pc/models/modelscope')
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
encoder = load_encoder(model)
result = {}
for name in ('inputs16k', 'inputs32k'):
    path = root/name/'prefill.json'
    data = json.loads(path.read_text())
    requests = data['requests']
    assert len(requests) == 16
    assert len({tuple(r['input_ids']) for r in requests}) == 16
    for r in requests:
        prompt = encoder([{'role': 'user', 'content': r['prompt']}], thinking_mode='chat')
        assert tokenizer.encode(prompt, add_special_tokens=False) == r['input_ids'], r['index']
        assert r['prompt_tokens'] == len(r['input_ids'])
        assert hashlib.sha256(json.dumps(r['input_ids']).encode()).hexdigest() == r['input_sha256']
    result[name] = dict(official_chat_encoding_exact=16, total_tokens=sum(r['prompt_tokens'] for r in requests),
                       manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), revision=data['revision'])
(root/'input-validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
