import importlib.util
from pathlib import Path

import pytest
import torch


def test_sparse_fixture_compaction_and_validation(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location(
        'fixture_writer', root / 'python/sglang/kernels/ops/debug/dsv4_tp8_sparse_fixture.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    q = torch.zeros((128, 8, 512), dtype=torch.bfloat16)
    kv = torch.arange(8 * 512).view(8, 512).to(torch.bfloat16)
    indices = torch.tensor([5, 1, 5, 7, -1], dtype=torch.int32)
    ptr = torch.full((129,), 4, dtype=torch.int32)
    ptr[:3] = torch.tensor([0, 2, 2])  # Includes an empty row, repeated slot.
    args = dict(q=q, kv=kv, indices=indices, indptr=ptr,
                sink=torch.arange(64, dtype=torch.float32), output=q,
                scale=512**-.5, provenance={'kind': 'CPU_test'})
    path = tmp_path / 'fixture.pt'
    payload = module.save_fixture(path, **args)
    assert torch.equal(payload['physical_slots'], torch.tensor([1, 5, 7]))
    assert torch.equal(payload['unified_kv'][payload['kv_indices'].long()],
                       kv[indices[:4].long()])
    assert torch.equal(payload['kv_indptr'], ptr)
    assert torch.equal(payload['attn_sink'], args['sink'][:8])
    restored = torch.load(path, weights_only=True)
    assert torch.equal(restored['q'], q)
    with pytest.raises(FileExistsError):
        module.save_fixture(path, **args)
    for bad in [torch.tensor([0, 2, 1] + [4]*126, dtype=torch.int32),
                torch.tensor([1] + [4]*128, dtype=torch.int32)]:
        with pytest.raises(ValueError, match='ragged'):
            module.save_fixture(tmp_path/'bad.pt', **(args | {'indptr': bad}))
    bad_ids = indices.clone()
    bad_ids[0] = -1
    with pytest.raises(ValueError, match='physical'):
        module.save_fixture(tmp_path/'bad.pt', **(args | {'indices': bad_ids}))
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda self: True))
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    with pytest.raises(RuntimeError, match='eager'):
        module.save_fixture(tmp_path/'graph.pt', **args)
    assert not (tmp_path/'graph.pt').exists()
