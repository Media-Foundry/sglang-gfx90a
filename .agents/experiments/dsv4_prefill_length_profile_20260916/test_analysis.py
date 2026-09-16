"""CPU-only analyzer coverage; synthetic duplicated frames are not GPU evidence."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parent
prior = root.parent / 'dsv4_premix_owner_20260916/profile'
original = [json.loads(p.read_text()) for p in sorted((prior/'markers').glob('rank-*-frame-*.json'))]
assert len(original) == 128
old_analysis = json.loads((prior/'analysis.json').read_text())
old_details = json.loads((prior/'details-analysis.json').read_text())
for n in (4, 8, 16):
    with tempfile.TemporaryDirectory(prefix='dsv4-profile-analysis-') as temporary:
        out = Path(temporary)
        (out/'markers').mkdir()
        for wave in range(4):
            for local in range(n):
                for rank in range(8):
                    old_seq = wave*4 + local%4 + 1
                    frame = copy.deepcopy(next(f for f in original if f['rank']==rank and f['sequence']==old_seq))
                    frame['sequence'] = wave*n + local + 1
                    (out/'markers'/f'rank-{rank}-frame-{frame["sequence"]:04d}.json').write_text(json.dumps(frame))
        (out/'complete.json').write_text(json.dumps(dict(input_echo_exact=64)))
        (out/'synthetic.stop.json').write_text(json.dumps(dict(remaining=[])))
        for wave in range(1,4):
            (out/f'trace{wave}-summary.json').write_text(json.dumps(dict(request_wall_s=old_analysis['waves'][wave-1]['http_s']*n/4)))
        command = [sys.executable,str(root/'analyze.py'),'--root',str(out),
                   '--stop-label','synthetic','--forwards-per-wave',str(n)]
        subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
        result=json.loads((out/'analysis.json').read_text())
        detail=json.loads((out/'details-analysis.json').read_text())
        assert result['snapshots']==n*4*8 and result['warmup_frames']==n*8
        assert len(result['selected_frames'])==n*3
        assert math.isclose(result['warm_mean_wave_ms'],old_analysis['warm_mean_wave_ms']*n/4,rel_tol=1e-12)
        for key,value in old_details['mean_wave_ms'].items():
            assert math.isclose(detail['mean_wave_ms'][key],value*n/4,rel_tol=1e-12)
        subprocess.run(command+['--check-only'],check=True,stdout=subprocess.DEVNULL)
        # Fail closed on an absent rank; equal-looking remaining frames are insufficient.
        (out/'markers/rank-7-frame-0001.json').unlink()
        failed=subprocess.run(command+['--check-only'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        assert failed.returncode!=0 and b'AssertionError' in failed.stderr
        print(f'PASS synthetic {n} forwards/wave: full closure, nested totals, missing-rank rejection')
