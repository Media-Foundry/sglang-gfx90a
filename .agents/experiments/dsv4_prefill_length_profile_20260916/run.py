"""Run length profiles serially only after both shorter-input regressions pass."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
accept=root.parent/'dsv4_prefill_mhc_common_regression_20260916/acceptance.json'
assert json.loads(accept.read_text())['status']=='passed_shorter_input_common_mhc_regression'
paths=[root/name for name in ('profile.py','analyze.py','run.py')]
frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
for length,n in (('8k',4),('16k',8),('32k',16)):
    assert frozen=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with (root/(length+'-driver.log')).open('x') as log:
        subprocess.run([sys.executable,str(root/'profile.py'),'--length',length],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(root/'analyze.py'),'--root',str(root/length),
                        '--stop-label','P'+length+'-markers-common','--forwards-per-wave',str(n)],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    print('PROFILE COMPLETE',length,flush=True)
summaries={}
for length in ('8k','16k','32k'):
    p=root/length
    a=json.loads((p/'analysis.json').read_text())
    d=json.loads((p/'details-analysis.json').read_text())
    count=json.loads((p/'complete.json').read_text())['input_tokens_per_wave']
    summaries[length]=dict(input_tokens=count,gpu_envelope_s=a['warm_mean_wave_ms']/1000,
        envelope_ms_per_token=a['warm_mean_wave_ms']/count,
        stages_ms_per_token={k:v/count for k,v in a['warm_mean_stages_ms'].items()},
        nested_ms_per_token={k:v/count for k,v in a['warm_mean_subdivisions_ms'].items()},
        details_ms_per_token={k:v/count for k,v in d['mean_wave_ms'].items()})
target=root/'length-comparison.json'
assert not target.exists()
target.write_text(json.dumps(dict(diagnostic_only=True,profiles=summaries,
    caveat='Nested scopes overlap; do not sum them with enclosing stages. Not formal throughput.'),indent=2)+'\n')
print(json.dumps(summaries,indent=2))
