"""Serial isolated stage1 counter passes with exact commands/logs retained."""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
assert json.loads((root/'timing.json').read_text())['captured_intermediate_byte_exact']
target=root/'collection.json';assert not target.exists()
env=dict(os.environ,HIP_VISIBLE_DEVICES='5',GPU_ARCHS='gfx90a',OMP_NUM_THREADS='1',
    ROCM_HOME='/opt/rocm',ROCM_PATH='/opt/rocm',
    PATH='/opt/rocm/bin:/home/pc/anaconda3/envs/DS/bin:/usr/local/bin:/usr/bin:/bin')
record=dict(status='running',passes=[])
def save():target.write_text(json.dumps(record,indent=2)+'\n')
save()
for name,counters in [('memory',['FetchSize','WriteSize']),
                      ('issue',['SQ_INSTS_MFMA','SQ_INSTS_VALU']),
                      ('lds',['LDSBankConflict'])]:
    output=root/name;assert not output.exists()
    result=root/(name+'-result.json')
    command=['/opt/rocm/bin/rocprofv3','--selected-regions','--kernel-trace',
        '--output-format','csv','--output-directory',str(output),'--pmc',*counters,'--',
        sys.executable,str(root/'probe.py'),'--profile','--output',str(result)]
    item=dict(name=name,command=command,counters=counters);record['passes'].append(item);save()
    print('COUNTERS',name,flush=True)
    with (root/(name+'.log')).open('x') as log:
        status=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    item['returncode']=status;save()
    if status:
        record['status']='failed';save();raise RuntimeError(f'{name} failed; inspect retained log')
    proof=json.loads(result.read_text());assert proof['profiled_exact_replays']==3
record['status']='complete';save()
