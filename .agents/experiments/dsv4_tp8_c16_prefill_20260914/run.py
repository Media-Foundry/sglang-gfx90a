"""One owned TP8 C16 prefill service arm; change only the chunk budget."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OLD = REPO / '.agents/experiments/dsv4_tp8_revalidation_20260913'
HELPER = REPO / '.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('A1', 'B', 'A2'), required=True)
    args = parser.parse_args()
    out = ROOT / args.arm
    out.mkdir(exist_ok=True)
    budget = 32768 if args.arm == 'B' else 36864
    spec = importlib.util.spec_from_file_location('owned_lifecycle', HELPER)
    life = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(life)
    life.ROOT = out
    life.OLD = out
    label = 'P16-' + args.arm
    if (out / f'{label}.state.json').exists():
        raise RuntimeError('Refuse to overwrite existing arm')

    original = (OLD / 'start-ar-matrix.sh').read_text()
    line = 'export CHUNKED_PREFILL_SIZE=36864 MAX_PREFILL_TOKENS=36864 PREFILL_MAX_REQUESTS=16'
    assert original.count(line) == 1
    generated = original.replace(line, f'export CHUNKED_PREFILL_SIZE={budget} MAX_PREFILL_TOKENS={budget} PREFILL_MAX_REQUESTS=16')
    (out / 'start-ar-matrix.sh').write_text(generated)
    manifest = json.loads((OLD / 'manifests64/prefill.json').read_text())
    manifest['requests'] = manifest['requests'][:16]
    assert len(manifest['requests']) == 16
    assert len({tuple(r['input_ids']) for r in manifest['requests']}) == 16
    (out / 'inputs.json').write_text(json.dumps(manifest, indent=2) + '\n')
    life.save('plan.json', dict(arm=args.arm, budget=budget,
        total_input_tokens=sum(len(r['input_ids']) for r in manifest['requests']),
        manifest_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest(),
        baseline_launcher_sha256=hashlib.sha256(original.encode()).hexdigest(),
        generated_launcher_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        lifecycle_helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest()))

    state = life.start(label, 0)
    try:
        life.ready(state)
        info = json.loads((out / f'{label}.server-info.json').read_text())
        assert info['chunked_prefill_size'] == budget
        assert info['max_prefill_tokens'] == budget
        assert info['max_total_tokens'] == 1048576
        proc = life.owned(state)
        life.resources('france-before', proc)
        france = json.loads((REPO/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
        answer = life.post(life.URL+'/generate', dict(input_ids=france['input_ids'],
            cache_salt=f'p16-france-{args.arm}-{time.time_ns()}',
            sampling_params=dict(temperature=0, max_new_tokens=32, ignore_eos=False)),600)
        life.save('France.json', answer)
        assert 'paris' in answer['text'].lower() and life.completion_ids(answer)

        legs = ['B1', 'B2'] if args.arm == 'B' else [args.arm]
        jobs = [('warmup', 1, 1), *[(leg, 3, 1) for leg in legs], ('quality', 2, 128)]
        records = []
        for name, rounds, tokens in jobs:
            life.resources(name+'-before', life.owned(state))
            target = out / f'{name}.json'
            assert not target.exists()
            offset = Path(state['log']).stat().st_size
            command = [sys.executable, str(REPO/'scripts/rocm/bench_dsv4_prefill_diverse_concurrent.py'),
                '--base-url', life.URL, '--inputs', str(out/'inputs.json'),
                '--request-count', '16', '--rounds', str(rounds), '--tokens', str(tokens),
                '--output', str(target)]
            print('MEASURE', args.arm, name, budget, flush=True)
            with (out / f'{name}.client.log').open('w') as log:
                subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=1800)
            result = json.loads(target.read_text())
            assert len(result['rounds']) == rounds
            for row in result['rounds']:
                assert row['cached_tokens'] == [0]*16
                assert row['completion_lengths'] == [tokens]*16
            record = dict(name=name, measured=name in legs, output_tokens=tokens,
                rates=[r['aggregate_input_tok_s'] for r in result['rounds']],
                median=result['median_input_tok_s'], log_start=offset,
                log_end=Path(state['log']).stat().st_size)
            records.append(record)
            life.save('progress.json', records)
            print('RESULT', json.dumps(record), flush=True)

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('/home/pc/models/modelscope', local_files_only=True)
        validated = 0
        for name, _, _ in jobs:
            result = json.loads((out / f'{name}.json').read_text())
            for row in result['rounds']:
                for ids, text in zip(row['completion_ids'], row['texts'], strict=True):
                    assert ids and tokenizer.decode(ids, skip_special_tokens=False) == text
                    validated += 1
        quality = json.loads((out/'quality.json').read_text())
        repeat_exact = sum(a==b for a,b in zip(quality['rounds'][0]['completion_ids'],
                                             quality['rounds'][1]['completion_ids'], strict=True))
        life.save('validation.json', dict(id_text_checked=validated,
            quality_repeat_full_ids_exact=repeat_exact, quality_requests=16,
            scope='Integrity and repeated-output checks, not a semantic or bitwise-equivalence proof.'))
        life.save('complete.json', dict(arm=args.arm, budget=budget, records=records))
    finally:
        life.stop(state)


if __name__ == '__main__':
    main()
