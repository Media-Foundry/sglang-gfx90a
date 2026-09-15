"""One real large-M MoE fixture, explicitly scoped by original-V4 layer/rank."""
import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch

_active = ContextVar('dsv4_ck_stage_capture', default=None)
_claimed = False


def current():
    return _active.get()


def selected_target():
    layer = int(os.getenv('SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_LAYER', '21'))
    rank = int(os.getenv('SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_RANK', '5'))
    prefix = int(os.getenv('SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_MIN_PREFIX', '0'))
    if not (0 <= layer < 43 and 0 <= rank < 8 and prefix >= 0):
        raise ValueError('CK capture requires V4 layer0..42, TP8 rank0..7, nonnegative prefix')
    return layer, rank, prefix


class Capture:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.records = {}

    def tensor(self, name, value):
        assert name not in self.records, name
        cpu = value.detach().contiguous().cpu()
        path = self.root/(name+'.pt')
        torch.save(cpu, path)
        self.records[name] = dict(shape=list(value.shape), dtype=str(value.dtype),
            is_shuffled=bool(getattr(value, 'is_shuffled', False)),
            sha256=hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)

    def info(self, name, value):
        assert name not in self.records, name
        self.records[name] = value


@contextmanager
def _scope(layer, batch):
    global _claimed
    directory = os.getenv('SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_DIR')
    if not directory or _claimed:
        yield
        return
    target_layer, target_rank, min_prefix = selected_target()
    if layer != target_layer:
        yield
        return
    from sglang.srt.layers.dsv4_prefill_experiments import mix_pair_active
    if not mix_pair_active():
        yield
        return
    from sglang.srt.distributed import get_tp_group
    if get_tp_group().rank_in_group != target_rank:
        yield
        return
    if max(batch.extend_prefix_lens_cpu, default=0) < min_prefix:
        yield
        return
    assert batch.forward_mode.name == 'EXTEND'
    assert not torch.cuda.is_current_stream_capturing()
    assert 8192 <= batch.input_ids.numel() <= 65536
    _claimed = True
    capture = Capture(directory)
    capture.tensor('input_ids', batch.input_ids)
    capture.tensor('positions', batch.positions)
    capture.info('provenance', dict(layer=layer, rank=target_rank,
        extend_lens=list(map(int, batch.extend_seq_lens_cpu)),
        prefix_lens=list(map(int, batch.extend_prefix_lens_cpu)),
        environment={k:v for k,v in os.environ.items()
                     if k.startswith(('SGLANG_DSV4_', 'AITER_DSV4_'))}))
    token = _active.set(capture)
    try:
        yield
        assert {'stage1_out', 'stage2_accum', 'stage2_out', 'weight13', 'weight2'} <= capture.records.keys()
        (capture.root/'manifest.json').write_text(json.dumps(capture.records, indent=2)+'\n')
        print(f'CK stage fixture complete: layer{layer} rank{target_rank}', flush=True)
    finally:
        _active.reset(token)


@contextmanager
def scope(layer, batch, parent):
    with parent, _scope(layer, batch):
        yield
