"""Resume the verified live A1 after warmup; do not recreate its process."""
import hashlib,importlib.util,json,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
from transformers import AutoTokenizer
import stable_lifecycle

root=Path(__file__).resolve().parent;repo=root.parents[2];out=root/'A1'
args=SimpleNamespace(arm='A1');candidate=diagnostic=False
plan=json.loads((out/'plan.json').read_text());sources=plan['sources'];paths=set(sources)
assert sources=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths}
helper=repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec=importlib.util.spec_from_file_location('resume_life',helper);life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
life.ROOT=life.OLD=out;stable_lifecycle.install(life)
state=stable_lifecycle.attach_verified(json.loads((out/'P16-ck-unique-A1.state.json').read_text()))
life.save('resume-state.json',state)
life.save('resume-provenance.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(stable_lifecycle.__file__))})
manifest=json.loads((out/'inputs.json').read_text())
assert not (out/'A1.json').exists() and not (out/'quality-0.json').exists()
source=(root/'run.py').read_text()
tail=source[source.index('    progress=[]'):]
tail=tail.replace('    progress=[]',"    progress=json.loads((out/'progress.json').read_text())",1)
line=next(line for line in tail.splitlines() if line.startswith('    legs='))
tail=tail.replace(line,"    legs=[('A1',3)]",1)
exec(compile('try:\n'+tail,str(root/'run.py')+' [verified A1 resume]','exec'))
