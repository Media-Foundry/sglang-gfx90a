"""Independently decode saved completion IDs; no GPU or service requests."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=root/'ar-matrix')
    parser.add_argument('--output', type=Path, default=root/'output-text-validation.json')
    args = parser.parse_args()
    matrix = args.matrix.resolve()
    tokenizer_path = Path('/home/pc/models/modelscope')
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    state = json.loads((matrix / 'state.json').read_text())
    checks = []
    for item in state['results']:
        path = matrix / Path(item['output']).name
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
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('PASS', result['total_responses'], 'responses across', len(checks), 'files;', state['status'])


if __name__ == '__main__':
    main()
