from pathlib import Path

import torch

from sglang.srt.speculative.dspark_components.dspark_sts import StsDataRecorder


def test_sts_recorder_flush_threshold_counts_samples(tmp_path: Path):
    stem = tmp_path / "sts"
    recorder = StsDataRecorder(path_stem=str(stem), gamma=2, flush_every=4)

    recorder.record(
        confidence_raw=torch.zeros((3, 2)),
        num_correct_drafts=torch.tensor([0, 1, 2]),
    )
    assert not (tmp_path / "sts.0.pt").exists()

    recorder.record(
        confidence_raw=torch.ones((2, 2)),
        num_correct_drafts=torch.tensor([1, 0]),
    )
    shard = torch.load(tmp_path / "sts.0.pt", map_location="cpu")
    assert shard["logits"].shape == (5, 2)
    assert shard["prefix_mask"].tolist() == [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ]
