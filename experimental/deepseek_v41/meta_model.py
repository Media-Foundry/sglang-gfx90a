"""Meta-device structural model for DeepSeek-V4.1 bring-up.

This is deliberately not an inference implementation.  It creates only
representative meta tensors and a layer graph, which is enough to catch config,
layer-count, and shape mistakes before the 510 GB checkpoint is complete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from .metadata import DeepSeekV41MetaConfig


class DeepSeekV41MetaLayer(nn.Module):
    """One structural layer with representative, payload-free tensors."""

    def __init__(self, config: DeepSeekV41MetaConfig, spec: dict[str, Any], device: str):
        super().__init__()
        self.layer_id = int(spec["layer_id"])
        self.compression_ratio = int(spec["compression_ratio"])
        self.has_indexer = bool(spec["has_indexer"])
        self.has_kv_source = bool(spec["has_kv_source"])
        self.has_engram = bool(spec["has_engram"])

        # FP4 routed weights are packed two values per byte in the source
        # checkpoint.  Keeping packed representative shapes here prevents the
        # common mistake of allocating a logical BF16-sized placeholder.
        packed_w13 = (config.moe_intermediate_size, config.hidden_size // 2)
        packed_w2 = (config.hidden_size, config.moe_intermediate_size // 2)
        self.register_buffer(
            "routed_w13_packed_meta",
            torch.empty(packed_w13, dtype=torch.uint8, device=device),
            persistent=False,
        )
        self.register_buffer(
            "routed_w2_packed_meta",
            torch.empty(packed_w2, dtype=torch.uint8, device=device),
            persistent=False,
        )
        if self.has_engram:
            self.register_buffer(
                "engram_row_meta",
                torch.empty((config.engram_head_dim,), dtype=torch.uint8, device=device),
                persistent=False,
            )


class DeepSeekV41MetaModel(nn.Module):
    """A zero-payload model graph for V4.1 shape and routing smoke tests."""

    def __init__(
        self,
        config: DeepSeekV41MetaConfig,
        *,
        device: str = "meta",
        tp_size: int = 1,
    ):
        super().__init__()
        errors = config.validate(tp_size=tp_size)
        if errors:
            raise ValueError("invalid DeepSeek-V4.1 metadata:\n- " + "\n- ".join(errors))
        self.meta_config = config
        self.tp_size = tp_size
        self.meta_device = torch.device(device)
        self.register_buffer(
            "embed_tokens_meta",
            torch.empty(
                (config.vocab_size, config.hidden_size),
                dtype=torch.bfloat16,
                device=device,
            ),
            persistent=False,
        )
        self.layers = nn.ModuleList(
            DeepSeekV41MetaLayer(config, spec, device)
            for spec in config.layer_specs()
        )
        self.register_buffer(
            "lm_head_meta",
            torch.empty(
                (config.vocab_size, config.hidden_size),
                dtype=torch.bfloat16,
                device=device,
            ),
            persistent=False,
        )

    @classmethod
    def from_json(
        cls, config_path: str | Path, *, device: str = "meta", tp_size: int = 1
    ) -> "DeepSeekV41MetaModel":
        return cls(
            DeepSeekV41MetaConfig.from_json(config_path),
            device=device,
            tp_size=tp_size,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Validate input shape and return a payload-free hidden-state shape."""
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"input_ids must be integer, got {input_ids.dtype}")
        batch, seq = input_ids.shape
        return torch.empty(
            (batch, seq, self.meta_config.hidden_size),
            dtype=torch.bfloat16,
            device=self.meta_device,
        )

    def structural_manifest(self) -> dict[str, Any]:
        return {
            "architecture": self.meta_config.architectures[0]
            if self.meta_config.architectures
            else None,
            "tp_size": self.tp_size,
            "backbone_layers": self.meta_config.num_hidden_layers,
            "total_layers_including_mtp": self.meta_config.total_transformer_layers,
            "layer_count": len(self.layers),
            "vocab_size": self.meta_config.vocab_size,
            "hidden_size": self.meta_config.hidden_size,
            "routed_experts": self.meta_config.n_routed_experts,
            "representative_routed_w13_packed_shape": list(
                self.layers[0].routed_w13_packed_meta.shape
            ),
            "representative_routed_w2_packed_shape": list(
                self.layers[0].routed_w2_packed_meta.shape
            ),
            "engram_layers": [
                layer.layer_id for layer in self.layers if layer.has_engram
            ],
        }


__all__ = ["DeepSeekV41MetaLayer", "DeepSeekV41MetaModel"]
