"""Nonblocking, single-stream readback for opt-in GPU timestamp diagnostics."""

import torch


class RealtimeTraceReadback:
    """Keep at most one pinned snapshot in flight; never wait on GPU progress.

    Call after graph replay on the graph's stream. Later replays on that same
    stream cannot overwrite the GPU markers before the queued copy finishes.
    This is deliberately not a multi-stream/pdmux snapshot protocol.
    """

    def __init__(self, markers):
        self.host = torch.empty_like(markers, device="cpu", pin_memory=True)
        self.event = torch.cuda.Event()
        self.sequence = None
        self.stream = None

    def sample(self, markers, sequence):
        stream = torch.cuda.current_stream(markers.device)
        if self.stream is not None and stream != self.stream:
            raise RuntimeError("Realtime trace readback requires one replay stream")
        self.stream = stream
        completed = None
        if self.sequence is not None:
            if not self.event.query():
                # Do not wait or reuse a host buffer that DMA still owns.
                return None
            completed = (self.sequence, self.host.tolist())
        self.host.copy_(markers.detach(), non_blocking=True)
        self.event.record(stream)
        self.sequence = sequence
        return completed
