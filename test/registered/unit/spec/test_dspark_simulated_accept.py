import torch

from sglang.srt.speculative.dspark_components.dspark_verify import (
    bonus_for_correct_len,
)


def test_bonus_is_reselected_after_simulated_correct_len_override():
    logits = torch.tensor(
        [
            [9.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0],
            [0.0, 0.0, 9.0, 0.0],
            [0.0, 0.0, 0.0, 9.0],
        ]
    )

    bonus = bonus_for_correct_len(
        target_logits=logits,
        correct_len=torch.tensor([0, 1]),
        bs=2,
        verify_num_draft_tokens=2,
    )

    assert bonus.tolist() == [0, 3]
