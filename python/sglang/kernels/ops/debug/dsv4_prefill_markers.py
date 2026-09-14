"""Default-off, nonblocking native TP8 prefill stage-marker collection.

No Kineto/rocprofiler interception. Snapshot completion is awaited only by a
background CPU writer, never by the submitting scheduler thread. The HIP
wall-clock frequency converts realtime ticks; GPU events independently check
the outer envelope. Cross-rank clocks are not assumed synchronized.
"""
from contextvars import ContextVar
from functools import wraps
import json
import logging
import os
from pathlib import Path
import queue
import threading
import time

import torch

DIR = 'SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR'
_active = ContextVar('dsv4_prefill_markers_active', default=False)


def active():
    return _active.get()


def report(ticks, gpu_ms, metadata):
    assert len(ticks) == 44 and all(len(row) == 32 for row in ticks)
    begin, end = ticks[43][:2]
    assert 0 < begin < end and gpu_ms > 0
    assert metadata['wall_clock_khz'] > 0
    us_per_tick = 1000 / metadata['wall_clock_khz']
    layers = []
    for layer, row in enumerate(ticks[:43]):
        valid = all(row[i] > 0 for i in range(8)) and all(row[i+1] >= row[i] for i in range(7))
        layers.append(dict(layer=layer, ticks=row, coarse_valid=valid,
            coarse_us=[(row[i+1]-row[i])*us_per_tick for i in range(7)] if valid else None,
            start_us=(row[0]-begin)*us_per_tick if row[0] else None,
            end_us=(row[7]-begin)*us_per_tick if row[7] else None))
    return dict(**metadata, gpu_frame_ms=gpu_ms, us_per_tick=us_per_tick,
                realtime_frame_ms=(end-begin)*us_per_tick/1000,
                event_envelope_ratio=gpu_ms*1000/((end-begin)*us_per_tick),
                frame_ticks=[begin,end], layers=layers,
                scope='Per-rank relative GPU stages, includes waits; not standalone kernel durations or benchmark throughput.')


def eligible(runner, batch):
    cfg=runner.model_config.hf_text_config
    ps=runner.ps
    spec=batch.spec_algorithm
    return (getattr(cfg,'model_type',None)=='deepseek_v4'
            and getattr(cfg,'num_hidden_layers',None)==43
            and getattr(cfg,'hidden_size',None)==4096
            and ps.tp_size==ps.attn_tp_size==8 and ps.moe_ep_size==1
            and ps.attn_cp_size==1
            and (spec is None or spec.is_none())
            and batch.forward_mode.is_extend_without_speculative()
            and 8192 <= batch.input_ids.shape[0] <= 65536)


class Collector:
    def __init__(self, runner, device):
        assert torch.version.hip and 'gfx90a' in torch.cuda.get_device_properties(device).gcnArchName
        assert os.getenv('SGLANG_DSV4_GFX90A_REALTIME_TRACE_GRAPH_ONLY','0')=='0'
        from sglang.kernels.ops.debug.gfx90a_realtime_marker import _jit_marker
        _jit_marker()  # Cold module work must precede clock calibration.
        self.matrix=torch.zeros((44,32),dtype=torch.uint64,device=device)
        self.wall_clock_khz=int(_jit_marker().wall_clock_khz(self.matrix.device.index))
        # Prime the FFI call binding as well as the compiled module.
        _jit_marker().run(self.matrix[43],31)
        self.rank=runner.ps.tp_rank
        layers=list(runner.model.model.layers)
        assert len(layers)==43
        for layer,row in zip(layers,self.matrix[:43]):
            layer._gfx90a_realtime_trace=row
            layer.self_attn._gfx90a_realtime_trace=row
            layer.self_attn.wo_b._gfx90a_output_realtime_trace=row
            layer.mlp._gfx90a_realtime_trace=row
        self.directory=Path(os.environ[DIR]);self.directory.mkdir(parents=True,exist_ok=True)
        self.sequence=0;self.error=None
        self.pending=queue.Queue(maxsize=16)
        threading.Thread(target=self._writer,daemon=True,name='dsv4-marker-writer').start()

    def _writer(self):
        while True:
            host,start,end,ready,metadata=self.pending.get()
            try:
                ready.synchronize()  # CPU worker only: scheduler remains free to submit.
                result=report(host.tolist(),start.elapsed_time(end),metadata)
                path=self.directory/f'rank-{self.rank}-frame-{metadata["sequence"]:04d}.json'
                assert not path.exists()
                tmp=path.with_suffix('.json.partial')
                tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path)
            except BaseException as error:
                self.error=repr(error)
                logging.getLogger(__name__).exception('DSV4 prefill marker writer failed')
            finally:
                self.pending.task_done()

    def begin(self, batch):
        if self.error:raise RuntimeError(self.error)
        self.sequence+=1
        stream=torch.cuda.current_stream(self.matrix.device)
        self.matrix.zero_()
        start=torch.cuda.Event(enable_timing=True)
        end=torch.cuda.Event(enable_timing=True)
        ready=torch.cuda.Event()
        # Materialize lazy event handles before the timed region. Re-recording
        # is on this same stream; only the final records are consumed.
        end.record();ready.record();start.record()
        from sglang.kernels.ops.debug.gfx90a_realtime_marker import gfx90a_realtime_marker
        gfx90a_realtime_marker(self.matrix[43],0)
        return start,end,ready,dict(rank=self.rank,sequence=self.sequence,
            wall_clock_khz=self.wall_clock_khz,
            rows=int(batch.input_ids.shape[0]),requests=int(batch.batch_size),
            extend_lens=[int(x) for x in batch.extend_seq_lens_cpu],
            prefix_lens=[int(x) for x in batch.extend_prefix_lens_cpu],
            stream=int(stream.cuda_stream),
            cpu_submit_begin_ns=time.perf_counter_ns())

    def end(self, token):
        start,end,ready,metadata=token
        assert torch.cuda.current_stream(self.matrix.device).cuda_stream==metadata['stream']
        from sglang.kernels.ops.debug.gfx90a_realtime_marker import gfx90a_realtime_marker
        gfx90a_realtime_marker(self.matrix[43],1)
        end.record()
        host=torch.empty((44,32),dtype=torch.uint64,pin_memory=True)
        host.copy_(self.matrix,non_blocking=True)
        ready.record()
        metadata['cpu_submit_end_ns']=time.perf_counter_ns()
        self.pending.put_nowait((host,start,end,ready,metadata))


def instrument(fn):
    # Disabled decoration is identity, preserving production call overhead.
    if not os.getenv(DIR):return fn

    @wraps(fn)
    def wrapped(self, forward_batch, *args, **kwargs):
        runner=self.model_runner
        if not eligible(runner,forward_batch):return fn(self,forward_batch,*args,**kwargs)
        collector=getattr(self,'_dsv4_prefill_marker_collector',None)
        if collector is None:
            collector=Collector(runner,forward_batch.input_ids.device)
            self._dsv4_prefill_marker_collector=collector
        scope=_active.set(True)
        try:
            token=collector.begin(forward_batch)
            result=fn(self,forward_batch,*args,**kwargs)
            collector.end(token)
            return result
        finally:
            _active.reset(scope)
    return wrapped
