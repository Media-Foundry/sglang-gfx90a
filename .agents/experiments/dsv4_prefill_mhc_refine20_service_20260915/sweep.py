"""Independent full20 / exact8+12 ABBA using the audited shared lifecycle."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
driver=root.parent/'dsv4_prefill_mhc_config_iters_20260915/run.py'
for arm in ('A1','B','A2'):
    assert not (root/arm).exists(),f'Refuse overwrite: {arm}'
    print('ARM START',arm,flush=True)
    with (root/(arm+'-run.log')).open('x') as log:
        subprocess.run([sys.executable,str(driver),'--arm',arm,'--comb-refine'],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    assert (root/arm/'complete.json').exists()
    print('ARM COMPLETE',arm,flush=True)
