"""CPU-only smoke tests for the experimental DeepSeek-V4.1 bring-up helpers."""

from __future__ import annotations

import json
import struct

import torch

from experimental.deepseek_v41.engram_host import (
    EngramHostTable,
    EngramLayerHostTable,
    EngramPrefetcher,
    EngramRowMapping,
    PinnedStagingPool,
    SafetensorsRowStore,
)
from experimental.deepseek_v41.metadata import DeepSeekV41MetaConfig, audit_checkpoint


def _config() -> dict:
    return {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "text_config": {
            "model_type": "deepseek_v41_text",
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1280,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "moe_intermediate_size": 2304,
            "n_routed_experts": 384,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "vocab_size": 129280,
            "max_position_embeddings": 1048576,
            "sliding_window": 128,
            "compress_ratios": [0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
            "num_nextn_predict_layers": 3,
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_topk": 512,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "engram_layer_ids": [1, 14],
            "engram_num_embeddings": [384006168, 384016682],
            "engram_max_ngram_size": 4,
            "engram_vocab_size": 16000000,
            "engram_n_heads": 8,
            "engram_head_dim": 256,
            "engram_compressed_vocab_size": 99092,
            "hc_sinkhorn_iters": 20,
        },
    }


def _write_safetensors(path, key: str, shape: list[int], payload: bytes) -> None:
    header = {
        key: {
            "dtype": "U8",
            "shape": shape,
            "data_offsets": [0, len(payload)],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_v41_config_and_tp_validation():
    config = DeepSeekV41MetaConfig.from_dict(_config())
    assert config.model_type == "deepseek_v41"
    assert config.total_transformer_layers == 43
    assert config.validate(tp_size=8) == []
    assert config.layer_specs()[1]["has_engram"]
    assert config.logical_expert_weight_shapes["w1"] == (2304, 5120)


def test_safetensors_rows_and_mapping(tmp_path):
    path = tmp_path / "rows.safetensors"
    _write_safetensors(path, "layers.1.engram.embed.weight", [4, 3], bytes(range(12)))
    store = SafetensorsRowStore.from_file(path, "layers.1.engram.embed.weight")
    assert store.read_rows_bytes([3, 1, 3]) == [
        bytes([9, 10, 11]),
        bytes([3, 4, 5]),
        bytes([9, 10, 11]),
    ]
    mapping = EngramRowMapping(global_rows=10, tp_size=4, rank=3)
    assert mapping.local_rows_with_padding == 3
    assert mapping.owner_rank(9) == 3
    assert mapping.rank_ranges() == ((0, 3), (3, 6), (6, 9), (9, 10))
    assert mapping.global_start == 9
    assert mapping.global_end == 10
    try:
        mapping.global_to_local([8])
    except IndexError:
        pass
    else:
        raise AssertionError("non-owned row was accepted")
    store.close()


def test_prefetch_deduplicates_without_gpu_table():
    class FakeStore:
        rows = 4
        row_bytes = 3
        dtype = "U8"

        def read_rows_bytes(self, row_ids):
            return [bytes([int(row_id), 7, 9]) for row_id in row_ids]

        def close(self):
            pass

    layer = EngramLayerHostTable(
        layer_id=1,
        stores={"embed.weight": FakeStore()},
        mapping=EngramRowMapping(global_rows=4),
    )
    table = EngramHostTable({1: layer})
    pool = PinnedStagingPool(slots=1, max_rows=4, row_bytes=3, pin_memory=False)
    with EngramPrefetcher(table, pool) as prefetcher:
        result = prefetcher.fetch_sync(1, [2, 1, 2])
        assert result.unique_row_ids == (2, 1)
        assert result.requested_to_unique == (0, 1, 0)
        assert result.host_tensor.tolist() == [[2, 7, 9], [1, 7, 9]]
        result.release()
    table.close()


def test_incomplete_checkpoint_is_not_called_ready(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    engram_keys = {
        f"layers.{layer}.engram.{name}": "model-00001-of-00001.safetensors"
        for layer in (1, 14)
        for name in (
            "embed.weight",
            "embed.scale",
            "q_weight",
            "k_weight",
            "wkv.weight",
            "wkv.scale",
        )
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": 12},
            "weight_map": engram_keys,
        })
    )
    report = audit_checkpoint(tmp_path, tp_size=8)
    assert report["status"] == "incomplete"
    assert report["counts"]["missing_shards"] == 1
    assert not report["errors"]
