"""Archive the successful preshuffle16 live check separately from failed v1."""
import hashlib
import json
from pathlib import Path
import re
import tarfile

root = Path(__file__).resolve().parent
directory = root / 'check-v2'
done = json.loads((directory / 'complete.json').read_text())
assert done['diagnostic'] and done['france_passed']
assert done['live_comparisons'] == 2688
assert not json.loads((directory / 'P32-owner-k32-check-v2.stop.json').read_text())['remaining']
log = (directory / 'P32-owner-k32-check-v2.service.log').read_text()
hits = re.findall(r'\[TP(\d+)\] owner-K32 logits selected: .* preshuffle=16 ', log)
assert set(hits) == set(map(str, range(8)))
assert 'owner-K32 fallback:' not in log
assert 'max_total_num_tokens=1048576' in log
oracle = json.loads((root / 'oracle-shuffle16.json').read_text())
assert oracle['status'] == 'complete'
assert oracle['contract'] == dict(shuffle=16, fp16=False, fnuz=False)
target = root / 'check-v2-evidence.tar.gz'
manifest = root / 'check-v2-manifest.json'
assert not target.exists() and not manifest.exists()
files = [p for p in sorted(directory.rglob('*'))
         if p.is_file() and p.suffix in ('.json', '.log', '.sh', '.patch')]
files += [root / name for name in ('check-v2-driver.log', 'oracle-shuffle16.log')]
with tarfile.open(target, 'w:gz') as archive:
    for path in files:
        archive.add(path, arcname=str(path.relative_to(root)))
record = dict(tested_source_commit='a630fc8cf3', k32_service_validated=True,
              throughput_score=False, live_comparisons=2688,
              sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
              bytes=target.stat().st_size,
              files=[dict(path=str(p.relative_to(root)),
                          sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
manifest.write_text(json.dumps(record, indent=2) + '\n')
print('LIVE CHECK ARCHIVED', len(files), target.stat().st_size)
