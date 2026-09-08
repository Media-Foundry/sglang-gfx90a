#!/usr/bin/env python3
"""Freeze public committed SGLang source into separate P/D request manifests."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from transformers import AutoTokenizer
from build_dsv4_prefill_diverse_manifest import FILES_AND_TASKS, load_encoder


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--revision', default='54b93c45c2')
    p.add_argument('--model', default='/home/pc/models/modelscope')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    revision = subprocess.check_output(['git', 'rev-parse', args.revision]).decode().strip()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    encoder = load_encoder(Path(args.model))
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision]).decode().splitlines()
    paths = sorted(x for x in paths if x.startswith('python/sglang/srt/') and x.endswith('.py'))
    sources = {}
    def source(path):
        if path not in sources:
            sources[path] = subprocess.check_output(['git', 'show', f'{revision}:{path}']).decode()
        return sources[path]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for phase, target in [('decode', 512), ('prefill', 8192)]:
        destination = args.output_dir / f'{phase}.json'
        if destination.exists():
            raise FileExistsError(destination)
        requests = []
        for index, (preferred, task) in enumerate(FILES_AND_TASKS):
            first = preferred if preferred in paths else paths[index]
            ordered = [first] + [x for x in paths[index:] + paths[:index] if x != first]
            snippets, provenance = [], []
            for path in ordered:
                text = source(path)
                snippets.append(f'\n# File: {path}\n{text}')
                provenance.append({'path': path, 'sha256': hashlib.sha256(text.encode()).hexdigest(),
                                   'url': f'https://github.com/Media-Foundry/sglang-gfx90a/blob/{revision}/{path}'})
                if sum(len(x) for x in snippets) > target * 12:
                    break
            body = ''.join(snippets)
            def encode(n):
                content = (f'Review the following actual SGLang source excerpt. {task}\n'
                           'Give a detailed technical explanation, then propose a concrete patch and regression tests. '
                           'Distinguish facts visible in this excerpt from assumptions.\n```python\n' + body[:n] + '\n```')
                prompt = encoder([{'role': 'user', 'content': content}], thinking_mode='chat')
                return content, tokenizer.encode(prompt, add_special_tokens=False)
            lo, hi = 0, len(body)
            best = encode(0)
            while lo <= hi:
                mid = (lo + hi) // 2
                value = encode(mid)
                if len(value[1]) <= target:
                    best = value
                    lo = mid + 1
                else:
                    hi = mid - 1
            assert target - 8 <= len(best[1]) <= target
            requests.append({'index': index, 'task': task, 'prompt': best[0], 'input_ids': best[1],
                             'prompt_tokens': len(best[1]), 'source_candidates': provenance,
                             'input_sha256': hashlib.sha256(json.dumps(best[1]).encode()).hexdigest()})
        assert len({tuple(x['input_ids']) for x in requests}) == 32
        destination.write_text(json.dumps({'format': 'dsv4-open-code-pd-v1', 'revision': revision,
                                          'phase': phase, 'target_tokens': target,
                                          'requests': requests}, indent=2) + '\n')
        print(destination, hashlib.sha256(destination.read_bytes()).hexdigest(), flush=True)


if __name__ == '__main__':
    main()
