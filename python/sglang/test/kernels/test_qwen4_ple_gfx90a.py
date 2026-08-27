import pytest
import torch

from sglang.kernels.ops.qwen4_ple import (
    can_fuse_qwen4_ngram_hash,
    fused_qwen4_ngram_hash,
)


def _shift_right_ignore_eos(tensor: torch.Tensor, n: int, eos: int) -> torch.Tensor:
    idx = torch.arange(tensor.shape[1], device=tensor.device, dtype=torch.long)
    eos_pos = torch.where(tensor == eos, idx, -1)
    previous_inclusive = torch.cummax(eos_pos, dim=1).values
    previous = torch.cat(
        [eos_pos.new_full((tensor.shape[0], 1), -1), previous_inclusive[:, :-1]],
        dim=1,
    )
    position = idx.unsqueeze(0) - (previous + 1)
    source = idx - n
    gathered = tensor.gather(
        1, source.clamp_min(0).unsqueeze(0).expand(tensor.shape[0], -1)
    )
    return torch.where(
        (position >= n) & (source.unsqueeze(0) >= 0),
        gathered,
        tensor.new_full((), eos),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
@pytest.mark.parametrize("batch", [1, 4, 16, 32])
def test_qwen4_ngram_hash_matches_eager_with_eos(batch):
    torch.manual_seed(21)
    eos = 151645
    contexts = torch.randint(
        0, 200000, (batch, 3), dtype=torch.long, device="cuda"
    )
    if batch >= 4:
        contexts[1, 0] = eos
        contexts[2, 1] = eos
        contexts[3, :] = eos
    multipliers = torch.tensor(
        [1000003, 2000003, 3000017], dtype=torch.long, device="cuda"
    )
    vocab_sizes = torch.tensor(
        [1009 + 2 * index for index in range(16)],
        dtype=torch.long,
        device="cuda",
    )
    offsets = torch.cumsum(
        torch.cat([vocab_sizes.new_zeros(1), vocab_sizes[:-1]]), dim=0
    )
    shifted = [contexts] + [
        _shift_right_ignore_eos(contexts, shift, eos) for shift in (1, 2)
    ]
    blocks = []
    for ngram in (2, 3):
        start = (ngram - 2) * 8
        mixed = shifted[0] * multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(
                mixed, shifted[position] * multipliers[position]
            )
        blocks.append(
            (
                torch.remainder(
                    mixed[:, -1:].unsqueeze(-1),
                    vocab_sizes[start : start + 8].view(1, 1, -1),
                )
                + offsets[start : start + 8].view(1, 1, -1)
            )[:, 0]
        )
    expected = torch.cat(blocks, dim=-1)
    assert can_fuse_qwen4_ngram_hash(
        contexts, multipliers, vocab_sizes, offsets
    )
    actual = fused_qwen4_ngram_hash(
        contexts, multipliers, vocab_sizes, offsets, eos
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
