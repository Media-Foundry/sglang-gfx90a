#!/usr/bin/env python3
"""Check this report's frozen inputs, numbers, references, and optional TeX log."""
import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def tex_inputs(root):
    """Resolve only local literal inputs used by this report, recursively."""
    seen = set()
    figures = set()

    def local(name, suffix):
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f'Nonportable TeX input: {name}')
        if not path.suffix:
            path = path.with_suffix(suffix)
        resolved = root / path
        if not resolved.is_file():
            raise ValueError(f'Missing TeX input: {path}')
        return path

    def visit(path):
        if path in seen:
            return
        seen.add(path)
        content = (root / path).read_text()
        for command, name in re.findall(
            r'\\(input|include|includegraphics)(?:\[[^\]]*\])?\{([^}]+)\}', content
        ):
            ref = local(name, '.pdf' if command == 'includegraphics' else '.tex')
            if command == 'includegraphics':
                figures.add(ref)
            else:
                visit(ref)

    visit(Path('main.tex'))
    return seen, figures


def check_log(root):
    content = (root / 'main.log').read_text(errors='replace')
    forbidden = [r'Citation .+ undefined', r'Reference .+ undefined',
                 r'There were undefined', r'Rerun to get', r'Label\(s\) may have changed',
                 r'Missing character:', r'Font .+ not loadable',
                 r'Overfull \\[hv]box', r'^!']
    failures = [line for line in content.splitlines()
                if any(re.search(pattern, line) for pattern in forbidden)]
    if failures:
        raise ValueError('TeX log failed checks:\n' + '\n'.join(failures))
    if not (root / 'main.pdf').is_file():
        raise ValueError('Missing compiled main.pdf')


def check(root=ROOT, compiled=False):
    tex, figures = tex_inputs(root)
    contents = '\n'.join((root / p).read_text() for p in tex)
    cited = {key.strip() for keys in re.findall(r'\\cite\w*\{([^}]+)\}', contents)
             for key in keys.split(',')}
    bibkeys = re.findall(r'\\bibitem\{([^}]+)\}', contents)
    if cited - set(bibkeys):
        raise ValueError(f'Undefined citation keys: {cited - set(bibkeys)}')
    if len(bibkeys) != len(set(bibkeys)):
        raise ValueError('Duplicate bibliography keys')
    labels = re.findall(r'\\label\{([^}]+)\}', contents)
    refs = {key.strip() for keys in re.findall(r'\\(?:c|C|page)?ref\{([^}]+)\}', contents)
            for key in keys.split(',')}
    if refs - set(labels) or len(labels) != len(set(labels)):
        raise ValueError(f'Missing/duplicate labels: {refs - set(labels)}')
    if 'fontspec' in contents or '\\setmainfont' in contents:
        raise ValueError('Unexpected host font dependency')
    data = json.loads((root / 'data/results.json').read_text())
    if [r['concurrency'] for r in data['matrix']] != [1, 2, 4, 8, 16, 32, 64]:
        raise ValueError('Formal concurrency matrix is incomplete')
    table = (root / 'generated/tp8_matrix.tex').read_text()
    for row in data['matrix']:
        for key in ('prefill', 'decode'):
            cell = row[key]
            if len(cell['rates']) != 3 or abs(statistics.median(cell['rates']) - cell['median']) > 1e-8:
                raise ValueError(f'Incorrect three-round median at C{row["concurrency"]}')
        expected = (f"{row['concurrency']} & {row['prefill']['median']:.2f} & "
                    f"{row['decode']['median']:.2f} & {row['decode']['http_aggregate_tok_s']:.2f}")
        if expected not in table:
            raise ValueError(f'Stale table row: {expected}')
    for name in ('down_consumer', 'c1_isolation'):
        report = data[name]
        a = statistics.mean(report['arms'][arm]['resident_tok_s'] for arm in ('A1', 'A2'))
        b = statistics.mean(report['arms'][arm]['resident_tok_s'] for arm in ('B1', 'B2'))
        if abs((b / a - 1) * 100 - report['gain_pct']) > 1e-8:
            raise ValueError(f'Incorrect ABBA summary: {name}')
    if len(figures) != 8:
        raise ValueError(f'Expected eight data figures, found {len(figures)}')
    if compiled:
        check_log(root)
    print(f'PASS: {len(tex)} TeX inputs, {len(figures)} figures, {len(cited)} cited keys; '
          'seven three-round P/D cells and independent ABBA summaries' +
          ('; final TeX log clean.' if compiled else '.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiled', action='store_true')
    args = parser.parse_args()
    check(compiled=args.compiled)
