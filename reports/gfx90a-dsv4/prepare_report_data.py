#!/usr/bin/env python3
"""Freeze audited local experiment summaries for a self-contained report build.

Run explicitly when refreshing results. Normal TeX/arXiv builds use the frozen
data/tables/figures and never access GPUs, a server, or a parent repository.
"""
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MAIN = REPO / '.agents/experiments/dsv4_tp8_revalidation_20260913'
PILOT = REPO / '.agents/experiments/dsv4_tp8_ar_down_consumer_20260914'


def verify_archive(directory, manifest):
    path = directory / manifest['archive']
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert path.stat().st_size == manifest['bytes'] and digest == manifest['sha256'], path
    expected = {entry.get('path', entry.get('name')): entry for entry in manifest['files']}
    checked = set()
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            assert member.name in expected and member.name not in checked, member.name
            entry = expected[member.name]
            with archive.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert member.size == entry['bytes'] and digest == entry['sha256'], member.name
            checked.add(member.name)
    assert checked == expected.keys()
    print(f'Verified {len(checked)} archived files in {directory.name}.')


def main():
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path.relative_to(REPO))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    formal = read(MAIN/'final-report.json')
    env = read(MAIN/'environment.json')
    pilot = read(PILOT/'summary.json')
    c1 = read(PILOT/'C1-summary.json')
    component = read(PILOT/'component.json')
    main_archive = read(MAIN/'evidence-index.json')
    pilot_archive = read(PILOT/'evidence-index.json')
    verify_archive(MAIN, main_archive)
    verify_archive(PILOT, pilot_archive)
    matrix = []
    for cell in formal['cells']:
        for phase in ('prefill', 'final_decode'):
            rec = cell[phase]
            assert len(rec['rates']) == 3
            assert abs(statistics.median(rec['rates']) - rec['median']) < 1e-8
            rawpath = MAIN/rec['artifact']
            if rawpath.exists():
                assert hashlib.sha256(rawpath.read_bytes()).hexdigest() == rec['sha256']
        matrix.append(dict(
            concurrency=cell['concurrency'],
            prefill=cell['prefill'], decode=cell['final_decode'],
            baseline_decode=cell['baseline_decode'],
        ))
    assert [r['concurrency'] for r in matrix] == [1, 2, 4, 8, 16, 32, 64]
    for report in (pilot, c1):
        a = statistics.mean(report['arms'][s]['resident_tok_s'] for s in ('A1','A2'))
        b = statistics.mean(report['arms'][s]['resident_tok_s'] for s in ('B1','B2'))
        assert abs((b/a-1)*100-report['gain_pct']) < 1e-8
    dspark_source = REPO/'.agents/memory/dsv4_tp8_c32_mhc_fusion_checkpoint_20260911.md'
    sources[str(dspark_source.relative_to(REPO))] = hashlib.sha256(dspark_source.read_bytes()).hexdigest()
    data = dict(
        revision='v0.3', as_of='2026-09-14', archived_checkpoint='3d73a75923',
        pilot_checkpoint='fb2e39fca9', matrix=matrix,
        c1_gemv_abba=formal['c1_woa_acceptance'],
        long8k_abba=formal['long_context_abba'],
        long8k_speedup=formal['long_context_resident_speedup'],
        down_consumer=pilot, c1_isolation=c1, down_component=component,
        graph_tiers=formal['observed_graph_tiers'],
        software=env['packages'], model_metadata=env['model_metadata'],
        archives=dict(formal={k:main_archive[k] for k in ('archive','bytes','sha256')},
                      pilot={k:pilot_archive[k] for k in ('archive','bytes','sha256')}),
        historical_dspark=[
            dict(family='Control', resident=1073.16, accept=2.687, low=1058, high=1082, waves=4),
            dict(family='Fusion / FP32', resident=1127.48, accept=2.732, low=1118, high=1138, waves=4),
            dict(family='Fusion / FP16', resident=1142.26, accept=2.698, low=1131, high=1160, waves=4),
        ],
        historical_dspark_note='September11 ledger summary; rounded min/max, not raw replicate samples; ignore_eos=True,512 outputs, not revalidated on September14.',
        source_sha256=sources,
    )
    (ROOT/'data').mkdir(exist_ok=True)
    (ROOT/'generated').mkdir(exist_ok=True)
    (ROOT/'data/results.json').write_text(json.dumps(data,indent=2)+'\n')
    rows = [r'% Generated from data/results.json by prepare_report_data.py.',
            r'\begin{tabular}{@{}rrrr@{}}', r'\toprule',
            r'C & Prefill input tok/s & Resident output tok/s & HTTP output tok/s \\',
            r'\midrule']
    for r in matrix:
        rows.append(f"{r['concurrency']} & {r['prefill']['median']:.2f} & "
                    f"{r['decode']['median']:.2f} & {r['decode']['http_aggregate_tok_s']:.2f} \\\\")
    rows.extend([r'\bottomrule', r'\end{tabular}'])
    (ROOT/'generated/tp8_matrix.tex').write_text('\n'.join(rows)+'\n')
    print('Frozen seven P/D tiers, three rounds/cell, separate ABBA arms and source hashes.')


if __name__ == '__main__':
    main()
