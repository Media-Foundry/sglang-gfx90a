"""Explicit dispatch adapter for the opt-in native TP8 M32 finalize experiment."""
from dataclasses import dataclass

from sglang.kernels.ops.moe.gfx90a_deferred_finalize import Gfx90aDeferredFinalize


@dataclass(frozen=True)
class DeferredDispatch:
    base: object
    gfx90a_defer_reduction: bool = True

    def __getattr__(self, name):
        return getattr(self.base, name)

    def _replace(self, **kwargs):
        return DeferredDispatch(self.base._replace(**kwargs))


def run_gfx90a_deferred(experts, hidden, topk, pre_quant_input=None):
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatcher

    assert isinstance(experts.dispatcher, StandardDispatcher)
    assert (experts.moe_tp_size, experts.moe_ep_size) == (8, 1)
    assert not experts.reduce_results and not experts._dwdp_bound
    assert tuple(hidden.shape) == (32, 4096)
    dispatched = experts.dispatcher.dispatch(hidden_states=hidden, topk_output=topk)
    assert dispatched.hidden_states_scale is None
    if pre_quant_input is not None:
        dispatched = dispatched._replace(hidden_states_pre_quant=pre_quant_input)
    combined = experts.run_moe_core(dispatch_output=DeferredDispatch(dispatched))
    # Standard combine is a passthrough here; do not send this object through
    # FusedMoE.forward_impl's Tensor slice/contiguous or an EP collective.
    result = combined.hidden_states
    assert isinstance(result, Gfx90aDeferredFinalize), 'deferred AIter selector missed'
    return result
