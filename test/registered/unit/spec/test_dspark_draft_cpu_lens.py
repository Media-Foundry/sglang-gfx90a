import torch

from sglang.srt.speculative.dspark_components.dspark_draft import (
    resolve_draft_backend_cpu_lens,
)


def test_dspark_draft_cpu_lens_are_not_preexpanded():
    prefix = torch.tensor([17, 29, 41], dtype=torch.int32)
    cpu_lens, final_sum = resolve_draft_backend_cpu_lens(
        prefix_lens_cpu=prefix,
        prefix_lens_sum=int(prefix.sum()),
        query_token_num=5,
    )

    assert cpu_lens is prefix
    assert torch.equal(cpu_lens, torch.tensor([17, 29, 41], dtype=torch.int32))
    assert final_sum == 17 + 29 + 41 + 3 * 5
