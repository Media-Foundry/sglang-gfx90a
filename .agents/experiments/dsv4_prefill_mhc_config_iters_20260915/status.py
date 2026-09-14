"""Read-only progress with PID birth checks; stale state is not a live process."""
import json
from pathlib import Path
import psutil
import argparse

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
root=p.parse_args().root
for arm in ('A1','B','A2'):
    out=root/arm
    path=out/f'P16-mhc20-{arm}.state.json'
    if not path.exists():
        print(json.dumps(dict(arm=arm,state='not-started')))
        continue
    state=json.loads(path.read_text())
    live=False
    try:
        proc=psutil.Process(state['pid'])
        live=abs(proc.create_time()-state['birth'])<.01 and proc.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:pass
    record=dict(arm=arm,pid=state['pid'],birth_verified_live=live,
                complete=(out/'complete.json').exists(),
                progress=json.loads((out/'progress.json').read_text()) if (out/'progress.json').exists() else [])
    if live:
        env=proc.environ()
        record.update(children=len(proc.children(recursive=True)),
                      config_iters_env=env.get('SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS'),
                      comb_refine_env=env.get('SGLANG_DSV4_PREFILL_MHC_COMB_REFINE20'),
                      wide_env=env.get('SGLANG_DSV4_C4_PREFILL_QUERY_WIDE'))
    stop=out/f'P16-mhc20-{arm}.stop.json'
    if stop.exists():record['remaining']=json.loads(stop.read_text())['remaining']
    print(json.dumps(record))
