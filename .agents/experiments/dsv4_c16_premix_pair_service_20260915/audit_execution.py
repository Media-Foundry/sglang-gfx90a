"""Verify recorded launcher isolation, all-rank hits and formal log intervals."""
import hashlib
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parent
output = root/'execution-audit.json'
assert not output.exists()
normalized = []
result = {'arms': {}, 'scope': 'First-hit rows are exact; scheduler token counts are page-rounded, not per-forward exact-M proof.'}
flag = 'SGLANG_DSV4_PREFILL_MIX_PAIR_COLUMNS'
for arm in ('A1', 'B', 'A2'):
    p = root/arm
    enabled = int(arm == 'B')
    launcher = (p/'start-ar-matrix.sh').read_text()
    line = f'export {flag}={enabled}\n'
    assert launcher.count(line) == 1
    normalized.append(launcher.replace(line, f'export {flag}=VALUE\n'))
    info = json.loads((p/f'P16-mix-pair-{arm}.server-info.json').read_text())
    assert info['tp_size'] == 8 and info['speculative_algorithm'] is None
    assert info['model_path'] == '/home/pc/models/modelscope'
    assert info['max_total_tokens'] == 1048576
    assert info['chunked_prefill_size'] == info['max_prefill_tokens'] == 32768
    assert 'paris' in json.loads((p/'France.json').read_text())['text'].lower()
    raw = (p/f'P16-mix-pair-{arm}.service.log').read_bytes()
    lines = raw.decode(errors='replace').splitlines()
    pairs = [line for line in lines if 'prefill mix-pair selected:' in line]
    if enabled:
        assert len(pairs) == 8
        for rank in range(8):
            assert any(f'TP{rank}]' in line and 'group=8 columns=2' in line for line in pairs)
    else:
        assert pairs == []
    for rank in range(8):
        assert any(f'TP{rank}]' in line and 'prefill mix-reuse4 selected' in line and 'group=8' in line for line in lines)
        assert any(f'TP{rank}]' in line and 'prefill query-reuse4 selected' in line and 'query_group=16' in line and 'runtime_m=1' in line for line in lines)
    intervals = []
    for record in json.loads((p/'progress.json').read_text()):
        if record['leg'] == 'warmup': continue
        text = raw[record['log_start']:record['log_end']].decode(errors='replace')
        warnings = [line for line in text.splitlines() if re.search('compil|ninja|Traceback|exception', line, re.I)]
        assert not warnings, warnings
        admissions = [line for line in text.splitlines() if 'Prefill batch,' in line]
        assert len(admissions) == 12
        assert all('#new-seq: 4,' in line and '#new-token: 32768,' in line and '#cached-token: 0,' in line for line in admissions)
        intervals.append({'leg': record['leg'], 'compile_or_exception_log_lines': warnings,
                          'page_rounded_admissions': admissions})
    result['arms'][arm] = {'pair_first_hits': pairs, 'formal_intervals': intervals,
        'service_log_sha256': hashlib.sha256(raw).hexdigest()}
assert normalized[0] == normalized[1] == normalized[2]
result['launcher_only_flag_difference'] = flag
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status': 'passed', 'arms': list(result['arms']), 'only_flag': flag}))
