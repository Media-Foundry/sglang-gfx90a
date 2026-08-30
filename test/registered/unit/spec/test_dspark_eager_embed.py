import torch

from sglang.srt.speculative.dspark_components.dspark_draft import (
    build_eager_draft_input_embeds,
)


def test_eager_embed_uses_model_specific_hc_expansion():
    class DraftModel:
        def forward_embed(self, input_ids):
            base = input_ids.to(torch.float32).view(-1, 1)
            return base.view(-1, 1, 1).expand(-1, 4, 8).clone()

    ids = torch.tensor([[3, 5], [7, 11]])
    out = build_eager_draft_input_embeds(
        draft_model=DraftModel(),
        embed_module=lambda _: (_ for _ in ()).throw(AssertionError("unused")),
        draft_block_ids=ids,
    )

    assert out.shape == (4, 4, 8)
    assert torch.equal(out[:, 0, 0], ids.reshape(-1).to(torch.float32))


def test_eager_embed_keeps_generic_flattened_lookup():
    ids = torch.tensor([[1, 2], [3, 4]])
    table = torch.arange(40, dtype=torch.float32).view(10, 4)
    out = build_eager_draft_input_embeds(
        draft_model=object(),
        embed_module=lambda token_ids: table[token_ids],
        draft_block_ids=ids,
    )

    assert out.shape == (4, 4)
    assert torch.equal(out, table[ids].reshape(4, 4))
