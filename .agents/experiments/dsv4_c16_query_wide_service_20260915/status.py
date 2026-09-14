"""Read-only experiment status: validate live PID birth, never infer it from a log."""
import json
from pathlib import Path

import psutil

root=Path(__file__).resolve().parent
for arm in ('A1','B','A2'):
    directory=root/arm
    path=directory/f'P16-wide16k-{arm}.state.json'
    if not path.exists():
        print(json.dumps(dict(arm=arm,state='not-started')))
        continue
    state=json.loads(path.read_text())
    live=False
    try:
        process=psutil.Process(state['pid'])
        live=(abs(process.create_time()-state['birth'])<.01
              and process.is_running() and process.status()!=psutil.STATUS_ZOMBIE
              and process.cmdline()==state['command'])
    except (psutil.NoSuchProcess,KeyError):pass
    progress=directory/'progress.json'
    result=dict(arm=arm,pid=state.get('pid'),birth_verified_live=live,
                complete=(directory/'complete.json').exists(),
                progress=json.loads(progress.read_text()) if progress.exists() else [])
    if live:
        result['children']=len(process.children(recursive=True))
        result['wide_env']=process.environ().get('SGLANG_DSV4_C4_PREFILL_QUERY_WIDE')
        result['query_reuse_env']=process.environ().get('SGLANG_DSV4_C4_PREFILL_QUERY_REUSE4')
        result['query_group_env']=process.environ().get('SGLANG_DSV4_C4_PREFILL_QUERY_GROUP_SIZE')
        result['runtime_m_env']=process.environ().get('SGLANG_DSV4_C4_PREFILL_QUERY_RUNTIME_M')
        result['mix_group_env']=process.environ().get('SGLANG_DSV4_PREFILL_MIX_GROUP_SIZE')
    stop=directory/f'P16-wide16k-{arm}.stop.json'
    if stop.exists():result['remaining']=json.loads(stop.read_text())['remaining']
    print(json.dumps(result))
