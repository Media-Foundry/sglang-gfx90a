"""Default-off loader for the independently built unique-slot CK experiment."""
from functools import lru_cache
import hashlib
import importlib.util
import json
import logging
from pathlib import Path


def eligible(m, t, k, block_m, kernel_name):
    return 8192 <= m <= 36864 and t == 6 and k == 256 and block_m == 64 and not kernel_name


@lru_cache(maxsize=1)
def load_verified(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['status'] != 'complete':
        raise ValueError('unique CK experiment requires a completed build')
    for path, digest in manifest['sources'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f'unique CK source changed since build: {path}')
    path = Path(manifest['module'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['module_sha256']:
        raise ValueError('unique CK binary hash mismatch')
    spec = importlib.util.spec_from_file_location(manifest['name'], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from sglang.srt.distributed import get_tp_group
    logging.getLogger(__name__).warning(
        'DSV4 unique-slot CK selected: rank=%s module=%s sha256=%s',
        get_tp_group().rank_in_group, manifest['name'], manifest['module_sha256'])
    return module
