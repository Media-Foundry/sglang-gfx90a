"""Read-only status; verify PID birth before calling an experiment live."""
import json
from pathlib import Path
import psutil
root=Path(__file__).resolve().parent
for arm in ('A1','B','A2'):
    directory=root/arm
    state_path=directory/f'P16-prefix-{arm}.state.json'
    if not state_path.exists():continue
    state=json.loads(state_path.read_text());live=False
    try:
        proc=psutil.Process(state['pid'])
        live=proc.create_time()==state['birth'] and proc.cmdline()==state['command']
    except (KeyError,psutil.NoSuchProcess):pass
    result=dict(arm=arm,pid=state.get('pid'),birth_verified_live=live,
                complete=(directory/'complete.json').exists())
    if (directory/'progress.json').exists():result['progress']=json.loads((directory/'progress.json').read_text())
    for name in ('warmup','A1','B1','B2','A2','quality'):
        path=directory/(name+'.json')
        if path.exists():
            data=json.loads(path.read_text())
            if data.get('status')!='complete':
                active=data.get('active_round',{})
                result['active_client']=dict(name=name,status=data.get('status'),
                    rounds_completed=len(data['rounds']),prime_saved='prime_responses' in active,
                    full_responses_saved='responses' in active)
    print(json.dumps(result))
