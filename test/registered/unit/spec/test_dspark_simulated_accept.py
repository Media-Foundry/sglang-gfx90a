from unittest.mock import patch

import torch

from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    TargetVerifyExecutor,
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


def test_dspark_accept_sync_rebuilds_commit_from_authoritative_decision():
    executor = TargetVerifyExecutor.__new__(TargetVerifyExecutor)
    executor.verify_num_draft_tokens = 4
    executor.gamma = 3
    executor._accept_sync_buf = None
    accept = AcceptOuts(
        correct_len=torch.tensor([0, 1]),
        bonus=torch.tensor([90, 91]),
        cap_trim_lens=torch.tensor([0, 0], dtype=torch.int32),
        commit_lens=torch.tensor([1, 2], dtype=torch.int32),
        new_seq_lens=torch.tensor([11, 22]),
        out_tokens=torch.zeros((2, 4), dtype=torch.int64),
    )

    class FakeGroup:
        world_size = 8

        @staticmethod
        def broadcast(tensor, src=0):
            assert src == 0 and tensor.is_contiguous() and tensor.numel() == 6
            tensor.copy_(torch.tensor([2, 0, 42, 43, 0, 0]))

    with (
        patch(
            "sglang.srt.speculative.dspark_components.dspark_verify."
            "envs.SGLANG_DSPARK_SYNC_ACCEPT_ACROSS_TP.get",
            return_value=True,
        ),
        patch(
            "sglang.srt.speculative.dspark_components.dspark_verify.get_tp_group",
            return_value=FakeGroup(),
        ),
    ):
        synced = executor.synchronize_accept_across_tp(
            accept=accept,
            draft_tokens=torch.tensor([[10, 11, 12], [20, 21, 22]]),
            prefix_lens=torch.tensor([100, 200]),
        )

    assert synced.correct_len.tolist() == [2, 0]
    assert synced.bonus.tolist() == [42, 43]
    assert synced.commit_lens.tolist() == [3, 1]
    assert synced.new_seq_lens.tolist() == [103, 201]
    assert synced.out_tokens.tolist() == [[10, 11, 42, 0], [43, 21, 22, 0]]
