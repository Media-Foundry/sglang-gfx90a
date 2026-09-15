import ast
from pathlib import Path


def test_only_producer_changes():
    tree=ast.parse(Path(__file__).with_name('run.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and
              any(isinstance(t,ast.Name) and t.id=='flags' for t in n.targets))
    def flags(candidate):
        env=dict(candidate=candidate,diagnostic=False)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'flags','exec'),env)
        return dict(line.removeprefix('export ').split('=',1) for line in env['flags'].splitlines()
                    if line.startswith('export '))
    a,b=flags(False),flags(True)
    assert {key for key in a if a[key]!=b[key]}=={'SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER'}
    assert a['SGLANG_DSV4_C4_PREFILL_QUERY_OWNER']==b['SGLANG_DSV4_C4_PREFILL_QUERY_OWNER']=='1'
    assert a['SGLANG_DSV4_DEBUG_OWNER_PRODUCER_CHECK']==a['SGLANG_DSV4_DEBUG_PREFILL_OWNER_CHECK']=='0'

    assert a['SGLANG_DSV4_PREFILL_ATTN_STAGE1']==b['SGLANG_DSV4_PREFILL_ATTN_STAGE1']=='1'
    assert a['SGLANG_DSV4_DEBUG_PREFILL_ATTN_STAGE1_CHECK']==b['SGLANG_DSV4_DEBUG_PREFILL_ATTN_STAGE1_CHECK']=='0'
