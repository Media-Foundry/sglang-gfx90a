import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

path=Path(__file__).with_name('stable_lifecycle.py')
spec=importlib.util.spec_from_file_location('stable_lifecycle_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_owned_uses_kernel_identity_not_realtime_birth(monkeypatch):
    value=dict(boot_id='boot',start_ticks=100)
    monkeypatch.setattr(module,'identity',lambda pid:value)
    process=SimpleNamespace(cmdline=lambda:['server'])
    monkeypatch.setattr(module.psutil,'Process',lambda pid:process)
    life=SimpleNamespace(start=lambda *args:None)
    module.install(life)
    state=dict(pid=123,kernel_identity=value.copy(),command=['server'],birth=-999)
    assert life.owned(state) is process
    value['start_ticks']=101
    with pytest.raises(AssertionError):life.owned(state)

def test_changed_command_is_not_owned(monkeypatch):
    value=dict(boot_id='boot',start_ticks=100)
    monkeypatch.setattr(module,'identity',lambda pid:value)
    monkeypatch.setattr(module.psutil,'Process',lambda pid:SimpleNamespace(cmdline=lambda:['other']))
    life=SimpleNamespace(start=lambda *args:None);module.install(life)
    with pytest.raises(AssertionError):life.owned(dict(pid=123,kernel_identity=value,command=['server']))
