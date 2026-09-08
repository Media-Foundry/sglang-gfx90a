"""Experimental paired-graph capture primitive; not enabled by any runner.

Keep a candidate graph/output pair in a separate pool without replacing the
baseline mappings. The caller owns distributed synchronization, capture scopes,
memory admission, and idle-only arm switching. No CUDA/Torch import is needed to
test the transaction's failure semantics.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CapturedAlternative:
    graph: Any
    output: Any
    pool: Any


def capture_alternative(
    backend: Any,
    key: Any,
    *,
    pool: Any,
    set_allocator_pool: Callable[[Any], None],
    capture: Callable[[], None],
) -> CapturedAlternative:
    """Capture exactly one alternate key, restoring baseline state on all exits.

    ``capture`` must use the backend's ordinary capture implementation (including
    warmup/reset hooks), not recursively invoke a paired-capture wrapper. Invoke
    this inside the communicator's outer capture/IPC registration scope. The
    returned pool must stay alive as long as its graph and output.
    """
    original_pool = backend._pool
    if pool is None or pool == original_pool:
        raise ValueError("candidate requires a distinct graph pool")
    # Fail before changing allocator state if the baseline is incomplete.
    original_graph = backend._graphs[key]
    original_output = backend._outputs[key]
    try:
        backend._pool = pool
        set_allocator_pool(pool)
        capture()
        candidate_graph = backend._graphs[key]
        candidate_output = backend._outputs[key]
        if candidate_graph is original_graph:
            raise RuntimeError("capture did not replace the requested graph")
        if candidate_output is original_output:
            raise RuntimeError("candidate reused the baseline output object")
        return CapturedAlternative(candidate_graph, candidate_output, pool)
    finally:
        backend._graphs[key] = original_graph
        backend._outputs[key] = original_output
        backend._pool = original_pool
        set_allocator_pool(original_pool)
