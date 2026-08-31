from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)


class _CountingMarkov:
    def __init__(self, gamma: int):
        self.gamma = gamma
        self.calls = 0

    def sample_block(
        self,
        base_logits,
        *,
        first_prev_tokens,
        hidden_states,
        sampler,
        collect_corrected=True,
    ):
        del first_prev_tokens, sampler
        self.calls += 1
        assert hidden_states.shape[:2] == base_logits.shape[:2]
        tokens = torch.arange(
            base_logits.shape[0] * self.gamma,
            device=base_logits.device,
            dtype=torch.int64,
        ).view(base_logits.shape[0], self.gamma)
        corrected = base_logits.clone() if collect_corrected else None
        return tokens, corrected

    def sample_block_tp_local_greedy(
        self, base_logits, *, first_prev_tokens, hidden_states
    ):
        del first_prev_tokens, hidden_states
        self.calls += 1
        return torch.full(
            base_logits.shape[:2], 7, dtype=torch.int64, device=base_logits.device
        )


class _FakeModel:
    def __init__(self, gamma: int, vocab: int):
        self.sample_from_anchor = True
        self.markov_head = _CountingMarkov(gamma)
        self.lm_head = SimpleNamespace(
            org_vocab_size=vocab,
            weight=torch.empty(vocab, 1, dtype=torch.bfloat16),
        )

    def compute_base_logits(self, hidden_states):
        return (
            torch.zeros(
                hidden_states.shape[0],
                self.lm_head.org_vocab_size,
                dtype=torch.bfloat16,
                device=hidden_states.device,
            ),
            None,
        )


@pytest.mark.parametrize("folded_sampling", [False, True])
def test_dspark_draft_sampler_runs_markov_block_once(folded_sampling):
    gamma, bs, vocab = 5, 3, 8
    model = _FakeModel(gamma, vocab)
    sampler = DsparkDraftSampler(
        model=model,
        gamma=gamma,
        max_bs=bs,
        device=torch.device("cpu"),
        folded_sampling=folded_sampling,
    )
    hidden = torch.zeros(bs * gamma, 4, dtype=torch.bfloat16)
    anchors = torch.arange(bs * gamma, dtype=torch.int64)

    sampler(hidden, anchors)

    assert model.markov_head.calls == 1
    assert sampler.out[: bs * gamma].tolist() == list(range(bs * gamma))
    if folded_sampling:
        assert torch.count_nonzero(sampler.corrected_out[: bs * gamma]) == 0


def test_dspark_draft_sampler_tp_local_greedy_calls_specialized_path(monkeypatch):
    gamma, bs, vocab = 2, 3, 8
    model = _FakeModel(gamma, vocab)
    monkeypatch.setenv("SGLANG_DSPARK_OPT_TP_LOCAL_GREEDY", "1")
    sampler = DsparkDraftSampler(
        model=model,
        gamma=gamma,
        max_bs=bs,
        device=torch.device("cpu"),
        folded_sampling=False,
    )
    hidden = torch.zeros(bs * gamma, 4, dtype=torch.bfloat16)
    anchors = torch.arange(bs * gamma, dtype=torch.int64)

    sampler(hidden, anchors)

    assert model.markov_head.calls == 1
    assert sampler.out[: bs * gamma].tolist() == [7] * (bs * gamma)
