from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark


def test_dspark_stages_disable_target_expert_recorder():
    active = False
    enters = 0

    @contextmanager
    def disabled_region():
        nonlocal active, enters
        assert not active
        active = True
        enters += 1
        try:
            yield
        finally:
            active = False

    class Stage:
        def __call__(self, positions, hidden_states, forward_batch):
            assert active
            assert forward_batch.marker == "target-owned-recorder"
            return hidden_states + 1

    model = object.__new__(DeepseekV4ForCausalLMDSpark)
    object.__setattr__(model, "stages", [Stage(), Stage(), Stage()])
    recorder = SimpleNamespace(disable_this_region=disabled_region)
    hidden = torch.zeros((2, 4, 8), dtype=torch.bfloat16)

    with patch(
        "sglang.srt.models.deepseek_v4_dspark."
        "get_global_expert_distribution_recorder",
        return_value=recorder,
    ):
        output = model.forward(
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([0, 1]),
            forward_batch=SimpleNamespace(marker="target-owned-recorder"),
            input_embeds=hidden,
        )

    assert enters == 1
    assert not active
    torch.testing.assert_close(output.hidden_states, hidden + 3)
