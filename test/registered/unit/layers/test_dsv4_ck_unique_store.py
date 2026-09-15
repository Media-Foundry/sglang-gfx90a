import json
import pytest
from sglang.kernels.ops.debug.dsv4_ck_unique_store import eligible,load_verified


def test_shape_contract():
    assert eligible(8192,6,256,64,'')
    assert eligible(32767,6,256,64,'')
    assert eligible(36864,6,256,64,'')
    for args in ((8191,6,256,64,''),(65536,6,256,64,''),(8192,6,512,64,''),
                 (8192,6,256,128,''),(8192,5,256,64,''),(8192,6,256,64,'other')):
        assert not eligible(*args)


def test_incomplete_build_rejected(tmp_path):
    p=tmp_path/'bad.json';p.write_text(json.dumps({'status':'failed'}))
    with pytest.raises(ValueError,match='completed build'):load_verified(str(p))


def test_changed_source_rejected_before_import(tmp_path):
    source=tmp_path/'source';source.write_text('changed')
    p=tmp_path/'bad.json';p.write_text(json.dumps({'status':'complete','sources':{str(source):'wrong'}}))
    with pytest.raises(ValueError,match='source changed'):load_verified(str(p))
