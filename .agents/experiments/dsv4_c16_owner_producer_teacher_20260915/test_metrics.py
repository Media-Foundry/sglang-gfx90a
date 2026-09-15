"""Check sign, count, overlap and tie-aware top1 calculations without GPU data."""
import ast
from pathlib import Path

import numpy as np
import pytest


def test_paired_statistics():
    tree=ast.parse(Path(__file__).with_name('analyze.py').read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    top=[[float(-i-1),i,None] for i in range(20)]
    changed=[row.copy() for row in top]
    changed[0][0],changed[1][0]=changed[1][0],changed[0][0]
    waves={'A':[dict(logprobs=[-.1,-.2],top20=[top,top])],
           'B':[dict(logprobs=[-.15,-.19],top20=[top,changed])]}
    context=dict(np=np,waves=waves)
    exec(compile(ast.Module(body=functions,type_ignores=[]),'metrics','exec'),context)
    compare=context['compare']
    same=compare('A','A')
    assert same['mean_nll_increase']==0 and same['target_logprob_exact']==2
    assert same['top20_exact_rows']==2 and same['top1_agreement']==1
    result=compare('A','B')
    assert result['mean_nll_increase']==pytest.approx(.02)
    assert result['target_logprob_abs_delta']['maximum']==pytest.approx(.05)
    assert result['positions']==2 and result['target_logprob_exact']==0
    assert result['top20_overlap']['mean']==1 and result['top20_exact_rows']==1
    assert result['top1_agreement']==.5 and len(result['top1_flips'])==1
    assert result['top1_flips'][0]['position']==1
    assert result['top1_flips'][0]['left_margin']==1
