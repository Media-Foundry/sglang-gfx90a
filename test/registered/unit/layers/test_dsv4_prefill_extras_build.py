"""Pure CPU checks for the offline builder's guarded, non-mutating overlays."""
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[4]/'scripts/rocm/build_dsv4_prefill_extras.py'
spec = importlib.util.spec_from_file_location('prefill_extras_build', SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def fixture(tmp_path):
    aiter = tmp_path/'aiter'
    base = '3rdparty/composable_kernel/include/'
    device = base+'ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp'
    scatter = base+'ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v7r3_scatter.hpp'
    files = {
        device: 'IsInputGemm ? InMemoryDataOperationEnum::Set : InMemoryDataOperationEnum::AtomicAdd;',
        scatter: '                dst_bufs(i).template Update<DstInMemOp, dst_vector_t>(\n'
                 '                    dst_offset, is_dst_valid, dst_vectors[i].template AsType<dst_vector_t>()[I0]);',
    }
    for name, text in files.items():
        path = aiter/name;path.parent.mkdir(parents=True, exist_ok=True);path.write_text(text)
    contract = {'aiter_contract': {name:hashlib.sha256(text.encode()).hexdigest() for name,text in files.items()}}
    return aiter, files, contract


def test_overlay_preserves_installed_headers_and_predicates_set(tmp_path):
    aiter, files, contract = fixture(tmp_path)
    outputs = builder.overlays(aiter,tmp_path/'overlay',contract)
    assert all((aiter/name).read_text()==text for name,text in files.items())
    assert 'AtomicAdd;' not in outputs[0].read_text()
    scatter = outputs[1].read_text()
    assert 'if(is_dst_valid)' in scatter
    assert 'dst_offset, true,' in scatter
    assert 'dst_offset, is_dst_valid,' in scatter  # non-Set operations unchanged


def test_unknown_header_fails_before_overlay_write(tmp_path):
    aiter, files, contract = fixture(tmp_path)
    (aiter/next(iter(files))).write_text('changed upstream')
    with pytest.raises(ValueError,match='Unsupported AIter/CK contract'):
        builder.overlays(aiter,tmp_path/'overlay',contract)
    assert not (tmp_path/'overlay').exists()


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    out = tmp_path/'build';out.mkdir();sentinel=out/'keep';sentinel.write_text('existing data')
    monkeypatch.setattr(sys,'argv',[str(SCRIPT),'--aiter-root',str(tmp_path/'absent'),'--output-dir',str(out)])
    with pytest.raises(FileExistsError):builder.main()
    assert sentinel.read_text()=='existing data'
