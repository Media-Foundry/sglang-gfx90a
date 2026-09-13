"""Independently decode saved completion IDs; no GPU or service requests."""
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    root = Path(__file__).resolve().parent
    tokenizer_path = Path('/home/pc/models/modelscope')
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    state = json.loads((root / 'ar-matrix/state.json').read_text())
    checks = []
    for item in state['results']:
        path = root / 'ar-matrix' / Path(item['output']).name
        data = json.loads(path.read_text())
        entries = []
        for row in data['rounds']:
            if item['phase'] == 'decode':
                for wave in row['waves']:
                    entries.extend((row['round'], wave['wave'], q['index'], q['output_ids'], q['text'])
                                   for q in wave['requests'])
            else:
                assert len(row['completion_ids']) == len(row['texts'])
                entries.extend((row['round'], 0, i, ids, text)
                               for i, (ids, text) in enumerate(zip(row['completion_ids'], row['texts'])))
        for rep, wave, index, ids, text in entries:
            decoded = tokenizer.decode(ids, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
            assert decoded == text, (path.name, rep, wave, index, repr(decoded), repr(text))
        checks.append(dict(file=str(path.relative_to(root)), responses=len(entries),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result = dict(matrix_status=state['status'], total_responses=sum(x['responses'] for x in checks),
                  tokenizer_json_sha256=hashlib.sha256((tokenizer_path/'tokenizer.json').read_bytes()).hexdigest(),
                  checks=checks, caveat='ID/text integrity, not model numerical or factual correctness.')
    (root / 'output-text-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PASS', result['total_responses'], 'responses across', len(checks), 'files;', state['status'])


if __name__ == '__main__':
    main()
