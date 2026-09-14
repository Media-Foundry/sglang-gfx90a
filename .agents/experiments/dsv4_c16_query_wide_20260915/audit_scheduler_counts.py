"""Verify that scheduler new-token logs are page-rounded budget accounting."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

root=Path(__file__).resolve().parent
repo=root.parents[2]
policy=repo/'python/sglang/srt/managers/schedule_policy.py'
reporter=repo/'python/sglang/srt/managers/scheduler_components/metrics_reporter.py'
source=policy.read_text()
tree=ast.parse(source)
ceil=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='ceil_paged_tokens')
update=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_update_prefill_budget')
first=update.body[0]
assert isinstance(first,ast.Assign) and isinstance(first.value,ast.Call)
assert ast.unparse(first)=='extend_input_len = self.ceil_paged_tokens(extend_input_len)'
assert any(isinstance(n,ast.AugAssign) and ast.unparse(n)=='self.log_input_tokens += extend_input_len' for n in ast.walk(update))
assert 'log_input_tokens=adder.log_input_tokens' in reporter.read_text()
assert '#new-token: {prefill_stats.log_input_tokens}' in reporter.read_text()
namespace={}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ceil],type_ignores=[])),str(policy),'exec'),namespace)
round_up=lambda n:namespace['ceil_paged_tokens'](SimpleNamespace(page_size=256),n)
examples=[dict(actual_tokens=n,logged_budget_tokens=round_up(n)) for n in (1,510,16383,16384,32767,32768)]
assert round_up(32767)==round_up(32768)==32768
manifests={}
for name in ('inputs16k','inputs32k'):
    path=root/name/'prefill.json'
    data=json.loads(path.read_text())
    lengths=[len(r['input_ids']) for r in data['requests']]
    manifests[name]=dict(input_lengths=lengths,actual_token_sum=sum(lengths),
                         page_rounded_sum=sum(map(round_up,lengths)),
                         input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
result=dict(scope=__doc__,examples=examples,manifests=manifests,
    source_sha256={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (policy,reporter)},
    ceil_source=ast.get_source_segment(source,ceil),
    interpretation='Equal scheduler new-token histograms establish equal page-budget accounting, not identical actual model M, row placement, request grouping, or kernel dispatch. Client throughput uses actual input IDs, so the measured rate is unaffected.')
(root/'scheduler-accounting-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
