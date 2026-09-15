"""Execute the actual local-sink helper and unified call site without a GPU model."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[4] / 'python/sglang/srt/models/deepseek_v4.py'


def extracted():
    tree = ast.parse(SOURCE.read_text())
    base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MqaAttentionBase')
    method = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == '_local_attn_sink')
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), scope)
    # Select by the distinctive unified call rather than line numbers.
    branches = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                and isinstance(n.test, ast.Name) and n.test.id == 'unified_kv'
                and any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                        and c.func.attr == 'forward'
                        and any(k.arg == 'attn_sink' for k in c.keywords) for c in ast.walk(n))]
    assert len(branches) == 1
    return scope['_local_attn_sink'], branches[0]


@pytest.mark.parametrize('tp', [1, 2, 4, 8])
def test_unified_sink_tracks_query_head_ownership(tp):
    local_sink, branch = extracted()
    full = torch.linspace(-2, 2, 64, dtype=torch.float32)
    heads = 64 // tp
    outputs = []
    for rank in range(tp):
        obj = SimpleNamespace(attn_tp_size=tp, attn_tp_rank=rank, n_local_heads=heads,
                              n_heads=64, attn_sink=full, _attn_sink_local=None,
                              attn_mqa=object(), compress_ratio=4)
        sink = local_sink(obj)
        calls = []
        def forward(**kwargs):
            calls.append(kwargs)
            return kwargs['q']
        q = torch.zeros(1, heads, 512, dtype=torch.bfloat16)
        kv = torch.zeros(1, 512, dtype=torch.bfloat16)
        env = dict(self=obj, unified_kv=True, q_out=q, q=q, attn_k=kv, kv=kv,
                   attn_backend=SimpleNamespace(forward=forward), forward_batch=object(),
                   attn_sink=sink, inverse_rope_freqs=None, inverse_rope_positions=None)
        exec(compile(ast.Module(body=[branch], type_ignores=[]), str(SOURCE), 'exec'), env)
        assert len(calls) == 1
        consumed = calls[0]['attn_sink'][:heads]
        assert torch.equal(consumed, full[rank*heads:(rank+1)*heads])
        assert local_sink(obj).data_ptr() == sink.data_ptr()
        # One K with score0, V1: per-head sink contributes only to denominator.
        outputs.append(1 / (1 + consumed.exp()))
    assert torch.equal(torch.cat(outputs), 1 / (1 + full.exp()))
    if tp > 1:
        wrong = (1 / (1 + full[:heads].exp())).repeat(tp)
        assert not torch.equal(wrong, torch.cat(outputs))
