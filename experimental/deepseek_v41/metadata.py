"""Pure-Python metadata and checkpoint audit helpers for DeepSeek-V4.1.

The V4.1 repository is large and is downloaded shard by shard.  This module only
reads JSON and safetensors headers; it never materializes tensor payloads.  It is
therefore safe to run against a partially downloaded checkpoint and reports an
incomplete state instead of silently treating missing tensors as zeros.
"""

from __future__ import annotations

import json
import math
import re
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_SHARD_RE = re.compile(r"model-(\d+)-of-(\d+)\.safetensors$")
_EXPERT_RE = re.compile(r"(?:mlp|ffn)\.experts\.(\d+)\.")
_ENGRAM_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.engram\.(.+)$")
_MTP_RE = re.compile(r"(?:^|\.)mtp\.(\d+)\.")

# Safetensors stores most dtypes in one byte per element for the packed formats
# used by V4.1.  Unknown dtypes are kept as ``None`` rather than guessed.
DTYPE_BYTES: dict[str, int | None] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F4_E2M1": 1,
    "F4_E2M1FN": 1,
    "F8_E8M0": 1,
    "F8_E8M0FNU": 1,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "U16": 2,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
}


class MetadataError(ValueError):
    """Raised when a metadata file is malformed."""


@dataclass(frozen=True)
class SafetensorsTensorHeader:
    """Location and shape information for one safetensors tensor.

    ``data_start`` and ``data_end`` are absolute file offsets.  Keeping these
    offsets lets the Engram host reader mmap individual rows without loading a
    complete shard.
    """

    key: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int
    path: Path

    @property
    def element_bytes(self) -> int | None:
        return DTYPE_BYTES.get(self.dtype)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int | None:
        itemsize = self.element_bytes
        return None if itemsize is None else self.numel * itemsize

    @property
    def row_bytes(self) -> int | None:
        if not self.shape:
            return None
        itemsize = self.element_bytes
        return None if itemsize is None else math.prod(self.shape[1:]) * itemsize


def _read_header_object(path: Path) -> tuple[dict[str, Any], int]:
    try:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise MetadataError(f"{path}: truncated safetensors header length")
            (header_len,) = struct.unpack("<Q", prefix)
            # A corrupt/incomplete download should fail quickly rather than
            # attempting to allocate an arbitrarily large buffer.
            if header_len <= 0 or header_len > 256 * 1024 * 1024:
                raise MetadataError(f"{path}: invalid header length {header_len}")
            raw_header = stream.read(header_len)
            if len(raw_header) != header_len:
                raise MetadataError(f"{path}: truncated safetensors header body")
    except OSError as exc:
        raise MetadataError(f"{path}: cannot read header: {exc}") from exc

    try:
        header = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise MetadataError(f"{path}: invalid safetensors header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise MetadataError(f"{path}: safetensors header is not an object")
    return header, 8 + header_len


def read_safetensors_header(path: str | Path) -> dict[str, SafetensorsTensorHeader]:
    """Read tensor headers and absolute byte ranges without touching payloads."""

    path = Path(path)
    header, payload_start = _read_header_object(path)
    file_size = path.stat().st_size
    tensors: dict[str, SafetensorsTensorHeader] = {}
    for key, info in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(info, dict):
            raise MetadataError(f"{path}: metadata for {key!r} is not an object")
        dtype = info.get("dtype")
        shape = info.get("shape")
        offsets = info.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list):
            raise MetadataError(f"{path}: malformed metadata for {key!r}")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise MetadataError(f"{path}: malformed data_offsets for {key!r}")
        try:
            shape_tuple = tuple(int(dim) for dim in shape)
            begin, end = (int(offsets[0]), int(offsets[1]))
        except (TypeError, ValueError) as exc:
            raise MetadataError(f"{path}: non-integer shape/offset for {key!r}") from exc
        if any(dim < 0 for dim in shape_tuple) or begin < 0 or end < begin:
            raise MetadataError(f"{path}: invalid shape/offset for {key!r}")
        absolute_begin = payload_start + begin
        absolute_end = payload_start + end
        if absolute_end > file_size:
            raise MetadataError(
                f"{path}: payload range for {key!r} exceeds file size "
                f"({absolute_end} > {file_size})"
            )
        tensors[key] = SafetensorsTensorHeader(
            key=key,
            dtype=dtype,
            shape=shape_tuple,
            data_start=absolute_begin,
            data_end=absolute_end,
            path=path,
        )
    return tensors


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _pick(top: Mapping[str, Any], text: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in text:
            return text[name]
        if name in top:
            return top[name]
    return default


@dataclass(frozen=True)
class DeepSeekV41MetaConfig:
    """Small, dependency-free projection of the V4.1 text configuration."""

    architectures: tuple[str, ...]
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    num_nextn_predict_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    vocab_size: int
    max_position_embeddings: int
    sliding_window: int
    compress_ratios: tuple[int, ...]
    index_source_layer_ids: tuple[int, ...]
    kv_source_layer_ids: tuple[int, ...]
    index_topk: int
    index_n_heads: int
    index_head_dim: int
    engram_layer_ids: tuple[int, ...]
    engram_num_embeddings: tuple[int, ...]
    engram_max_ngram_size: int
    engram_vocab_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_compressed_vocab_size: int
    hc_sinkhorn_iters: int
    raw: Mapping[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "DeepSeekV41MetaConfig":
        top = _as_mapping(config)
        text = _as_mapping(top.get("text_config"))
        architectures = tuple(str(x) for x in top.get("architectures", ()))

        def integer(*names: str, default: int = 0) -> int:
            value = _pick(top, text, *names, default=default)
            return int(value)

        def integers(*names: str) -> tuple[int, ...]:
            value = _pick(top, text, *names, default=())
            if value is None:
                return ()
            return tuple(int(x) for x in value)

        return cls(
            architectures=architectures,
            # The HF wrapper uses ``deepseek_v41`` at the top level and
            # ``deepseek_v41_text`` inside ``text_config``.  Prefer the
            # top-level value for architecture dispatch while accepting both
            # spellings during validation.
            model_type=str(top.get("model_type", text.get("model_type", ""))),
            hidden_size=integer("hidden_size", "dim"),
            num_hidden_layers=integer("num_hidden_layers", "n_layers"),
            num_nextn_predict_layers=integer(
                "num_nextn_predict_layers", "n_mtp_layers", default=0
            ),
            num_attention_heads=integer("num_attention_heads", "n_heads"),
            num_key_value_heads=integer("num_key_value_heads", "n_kv_heads", default=1),
            head_dim=integer("head_dim", default=0),
            qk_rope_head_dim=integer("qk_rope_head_dim", default=0),
            q_lora_rank=integer("q_lora_rank", default=0),
            o_lora_rank=integer("o_lora_rank", default=0),
            o_groups=integer("o_groups", default=1),
            moe_intermediate_size=integer("moe_intermediate_size", "moe_inter_dim"),
            n_routed_experts=integer("n_routed_experts", "n_routed_experts"),
            n_shared_experts=integer("n_shared_experts", default=1),
            num_experts_per_tok=integer(
                "num_experts_per_tok", "n_active_experts", default=0
            ),
            vocab_size=integer("vocab_size", default=0),
            max_position_embeddings=integer("max_position_embeddings", "max_seq_len"),
            sliding_window=integer("sliding_window", "window_size", default=0),
            compress_ratios=integers("compress_ratios"),
            index_source_layer_ids=integers("index_source_layer_ids"),
            kv_source_layer_ids=integers("kv_source_layer_ids"),
            index_topk=integer("index_topk", default=0),
            index_n_heads=integer("index_n_heads", "indexer_n_heads", default=0),
            index_head_dim=integer("index_head_dim", "indexer_head_dim", default=0),
            engram_layer_ids=integers("engram_layer_ids"),
            engram_num_embeddings=integers("engram_num_embeddings"),
            engram_max_ngram_size=integer("engram_max_ngram_size", default=0),
            engram_vocab_size=integer("engram_vocab_size", default=0),
            engram_n_heads=integer("engram_n_heads", "engram_nheads", default=0),
            engram_head_dim=integer("engram_head_dim", default=0),
            engram_compressed_vocab_size=integer(
                "engram_compressed_vocab_size", default=0
            ),
            hc_sinkhorn_iters=integer("hc_sinkhorn_iters", "sinkhorn_iters", default=0),
            raw=config,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DeepSeekV41MetaConfig":
        with Path(path).open() as stream:
            value = json.load(stream)
        if not isinstance(value, Mapping):
            raise MetadataError(f"{path}: config JSON is not an object")
        return cls.from_dict(value)

    @property
    def total_transformer_layers(self) -> int:
        return self.num_hidden_layers + self.num_nextn_predict_layers

    @property
    def backbone_compress_ratios(self) -> tuple[int, ...]:
        return self.compress_ratios[: self.num_hidden_layers]

    @property
    def mtp_compress_ratios(self) -> tuple[int, ...]:
        return self.compress_ratios[self.num_hidden_layers :]

    @property
    def logical_expert_weight_shapes(self) -> dict[str, tuple[int, ...]]:
        """Logical (unpacked) routed-expert shapes used by the empty model."""
        return {
            "w1": (self.moe_intermediate_size, self.hidden_size),
            "w3": (self.moe_intermediate_size, self.hidden_size),
            "w2": (self.hidden_size, self.moe_intermediate_size),
        }

    def validate(self, tp_size: int = 1) -> list[str]:
        errors: list[str] = []
        if not any(arch == "DeepseekV41ForCausalLM" for arch in self.architectures):
            errors.append(
                "architectures does not contain DeepseekV41ForCausalLM "
                f"({list(self.architectures)!r})"
            )
        if self.model_type not in (
            "deepseek_v41",
            "deepseek_v4_1",
            "deepseek_v41_text",
            "deepseek_v4_1_text",
        ):
            errors.append(f"unexpected model_type={self.model_type!r}")
        for name in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "moe_intermediate_size",
            "n_routed_experts",
            "vocab_size",
        ):
            if getattr(self, name) <= 0:
                errors.append(f"{name} must be positive")
        if self.n_routed_experts and tp_size > 0 and self.n_routed_experts % tp_size:
            errors.append(
                f"n_routed_experts={self.n_routed_experts} is not divisible by tp_size={tp_size}"
            )
        if self.hidden_size % 2:
            errors.append("hidden_size must be even for packed FP4 expert weights")
        if self.moe_intermediate_size % 2:
            errors.append(
                "moe_intermediate_size must be even for packed FP4 expert weights"
            )
        if len(self.compress_ratios) != self.total_transformer_layers:
            errors.append(
                "compress_ratios length "
                f"{len(self.compress_ratios)} != backbone+nextn "
                f"{self.total_transformer_layers}"
            )
        if any(ratio < 0 for ratio in self.compress_ratios):
            errors.append("compress_ratios contains a negative value")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            errors.append("Engram layer_ids and num_embeddings have different lengths")
        if len(set(self.engram_layer_ids)) != len(self.engram_layer_ids):
            errors.append("Engram layer_ids are not unique")
        for layer_id in (
            *self.index_source_layer_ids,
            *self.kv_source_layer_ids,
            *self.engram_layer_ids,
        ):
            if layer_id < 0 or layer_id >= self.num_hidden_layers:
                errors.append(
                    f"layer id {layer_id} is outside the "
                    f"{self.num_hidden_layers}-layer backbone"
                )
        if self.index_topk <= 0:
            errors.append("index_topk must be positive")
        return errors

    def layer_specs(self) -> tuple[dict[str, Any], ...]:
        """Return a compact structural graph for all backbone and MTP layers."""
        index_layers = set(self.index_source_layer_ids)
        kv_layers = set(self.kv_source_layer_ids)
        engram_layers = set(self.engram_layer_ids)
        specs = []
        for layer_id, ratio in enumerate(self.compress_ratios):
            specs.append(
                {
                    "layer_id": layer_id,
                    "backbone": layer_id < self.num_hidden_layers,
                    "compression_ratio": ratio,
                    "has_indexer": layer_id in index_layers,
                    "has_kv_source": layer_id in kv_layers,
                    "has_engram": layer_id in engram_layers,
                }
            )
        return tuple(specs)


def converter_normalize_name(source_name: str) -> tuple[str, int | None]:
    """Mirror the name/dimension part of ``inference/convert.py``.

    This intentionally does not load a tensor or perform the converter's dtype
    conversion.  It is used only to audit whether the index names have a
    deterministic destination and which dimensions would be TP-sharded.
    """

    name = str(source_name)
    if name.startswith("model."):
        name = name[len("model.") :]
    name = name.replace("self_attn", "attn")
    if not name.startswith("vision."):
        name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    if any(
        marker in name
        for marker in ("hc", "attn_sink", "tie2eid", "tid2eid", "ape", "image_")
    ):
        key = name.split(".")[-1]
    else:
        components = name.split(".")
        key = components[-2] if len(components) >= 2 else components[-1]
    mapping = {
        "embed": ("embed", 0),
        "wq_b": ("wq_b", 0),
        "wo_a": ("wo_a", 0),
        "wo_b": ("wo_b", 1),
        "head": ("head", 0),
        "attn_sink": ("attn_sink", 0),
        "weights_proj": ("weights_proj", 0),
    }
    new_key, dim = mapping.get(key, (key, None))
    return name.replace(key, new_key), dim


def parameter_name_audit(keys: Iterable[str]) -> dict[str, Any]:
    """Return converter-normalization and expert-name diagnostics."""

    normalized_to_sources: dict[str, list[str]] = defaultdict(list)
    source_prefixes: dict[str, int] = defaultdict(int)
    malformed_expert_names: list[str] = []
    samples: list[dict[str, str | None]] = []
    for source_name in keys:
        source_name = str(source_name)
        normalized, dim = converter_normalize_name(source_name)
        normalized_to_sources[normalized].append(source_name)
        source_prefixes[source_name.split(".", 1)[0]] += 1
        if ".experts." in source_name and not _EXPERT_RE.search(source_name):
            malformed_expert_names.append(source_name)
        if len(samples) < 12 and normalized != source_name:
            samples.append(
                {
                    "source": source_name,
                    "normalized": normalized,
                    "tp_axis": str(dim) if dim is not None else None,
                }
            )
    collisions = {
        name: sorted(values)
        for name, values in normalized_to_sources.items()
        if len(values) > 1
    }
    return {
        "source_count": sum(source_prefixes.values()),
        "source_prefix_counts": dict(sorted(source_prefixes.items())),
        "normalized_collision_count": len(collisions),
        "normalized_collisions_preview": {
            name: values[:8] for name, values in list(sorted(collisions.items()))[:8]
        },
        "malformed_expert_name_count": len(malformed_expert_names),
        "malformed_expert_names_preview": malformed_expert_names[:8],
        "normalization_samples": samples,
    }


def tp_shard_plan(
    headers: Mapping[str, SafetensorsTensorHeader],
    weight_map: Mapping[str, Any],
    tp_size: int,
) -> dict[str, Any]:
    """Summarize the converter's per-tensor TP ownership from valid headers."""

    mode_counts: dict[str, int] = defaultdict(int)
    nondivisible: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for key, header in headers.items():
        normalized, axis = converter_normalize_name(key)
        if ".experts." in normalized:
            mode = "expert_owner"
            axis = None
        elif ".engram.embed." in normalized:
            mode = "dim0_ceil_pad"
            axis = 0
        elif _MTP_RE.search(normalized) and normalized.endswith(
            ("embed_tokens.weight", "head.weight")
        ):
            mode = "mtp_tied_skipped"
            axis = None
        elif axis is None:
            mode = "replicated_or_unsharded"
        else:
            mode = "uniform_dim"
            if tp_size <= 0 or header.shape[axis] % tp_size:
                nondivisible.append(
                    {
                        "key": key,
                        "normalized": normalized,
                        "axis": axis,
                        "size": header.shape[axis],
                    }
                )
        mode_counts[mode] += 1
        if len(samples) < 16:
            samples.append(
                {
                    "key": key,
                    "normalized": normalized,
                    "shape": list(header.shape),
                    "mode": mode,
                    "axis": axis,
                }
            )
    return {
        "tp_size": tp_size,
        "valid_header_tensors": len(headers),
        "mode_counts": dict(sorted(mode_counts.items())),
        "nondivisible_uniform_count": len(nondivisible),
        "nondivisible_uniform_preview": nondivisible[:16],
        "samples": samples,
    }


def _expected_shard_names(weight_map: Mapping[str, Any]) -> tuple[str, ...]:
    names = {str(value) for value in weight_map.values() if isinstance(value, str)}
    return tuple(sorted(names))


def _shard_sibling_incomplete(path: Path) -> Path | None:
    candidate = Path(str(path) + ".incomplete")
    return candidate if candidate.exists() else None


def _key_groups(keys: Iterable[str]) -> dict[str, Any]:
    engram: dict[int, list[str]] = {}
    mtp: dict[int, int] = {}
    vision = 0
    for key in keys:
        match = _ENGRAM_RE.search(key)
        if match:
            engram.setdefault(int(match.group(1)), []).append(match.group(2))
        match = _MTP_RE.search(key)
        if match:
            mtp[int(match.group(1))] = mtp.get(int(match.group(1)), 0) + 1
        if key.startswith("vision.") or ".vision." in key:
            vision += 1
    return {
        "engram": {str(layer): sorted(names) for layer, names in sorted(engram.items())},
        "mtp": {str(layer): count for layer, count in sorted(mtp.items())},
        "vision_count": vision,
    }


def audit_checkpoint(
    model_dir: str | Path,
    *,
    tp_size: int = 8,
    index_path: str | Path | None = None,
    validate_headers: bool = True,
) -> dict[str, Any]:
    """Audit a V4.1 checkpoint, tolerating an in-progress shard download.

    The returned dictionary is deliberately JSON serializable.  ``status`` is
    ``ready`` only when every indexed shard is present and every header is valid;
    it is ``incomplete`` for missing/partial payloads and ``invalid`` for a
    malformed config/index/header.
    """

    root = Path(model_dir)
    result: dict[str, Any] = {
        "model_dir": str(root),
        "tp_size": tp_size,
        "status": "invalid",
        "errors": [],
        "warnings": [],
        "files": {},
        "counts": {},
        "representative_tensors": {},
    }
    errors: list[str] = result["errors"]
    warnings: list[str] = result["warnings"]
    config_path = root / "config.json"
    chosen_index = Path(index_path) if index_path else root / "model.safetensors.index.json"
    if not config_path.exists():
        errors.append(f"missing {config_path.name}")
    if not chosen_index.exists():
        errors.append(f"missing {chosen_index.name}")
    if errors:
        return result

    try:
        with config_path.open() as stream:
            config = json.load(stream)
        with chosen_index.open() as stream:
            index = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read config/index: {exc}")
        return result
    if not isinstance(config, Mapping) or not isinstance(index, Mapping):
        errors.append("config/index must both be JSON objects")
        return result

    try:
        meta = DeepSeekV41MetaConfig.from_dict(config)
        errors.extend(meta.validate(tp_size=tp_size))
    except (TypeError, ValueError, KeyError) as exc:
        errors.append(f"invalid V4.1 config: {exc}")
        meta = None
    result["config"] = {
        "architectures": list(meta.architectures) if meta else config.get("architectures", []),
        "model_type": meta.model_type if meta else config.get("model_type"),
        "hidden_size": meta.hidden_size if meta else None,
        "num_hidden_layers": meta.num_hidden_layers if meta else None,
        "num_nextn_predict_layers": meta.num_nextn_predict_layers if meta else None,
        "n_routed_experts": meta.n_routed_experts if meta else None,
        "compress_ratios_length": len(meta.compress_ratios) if meta else None,
    }
    if meta:
        result["layer_specs"] = list(meta.layer_specs())

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        errors.append("index has no object-valued weight_map")
        return result
    expected_keys = {str(key) for key in weight_map}
    result["parameter_name_audit"] = parameter_name_audit(expected_keys)
    shard_names = _expected_shard_names(weight_map)
    result["counts"]["indexed_tensors"] = len(expected_keys)
    result["counts"]["indexed_shards"] = len(shard_names)
    shard_numbers = []
    declared_shard_count = None
    for shard_name in shard_names:
        match = _SHARD_RE.fullmatch(shard_name)
        if match:
            shard_numbers.append(int(match.group(1)))
            declared_shard_count = int(match.group(2))
    if shard_numbers and declared_shard_count is not None:
        expected_numbers = list(range(1, declared_shard_count + 1))
        if sorted(shard_numbers) != expected_numbers:
            errors.append(
                "indexed shard numbering is not contiguous: "
                f"found={sorted(shard_numbers)[:8]}... expected 1..{declared_shard_count}"
            )
    metadata = _as_mapping(index.get("metadata"))
    if "total_size" in metadata:
        result["counts"]["indexed_total_size"] = metadata["total_size"]

    complete_keys: set[str] = set()
    tensor_headers: dict[str, SafetensorsTensorHeader] = {}
    missing_shards: list[str] = []
    incomplete_shards: list[str] = []
    for shard_name in shard_names:
        path = root / shard_name
        file_info: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            sibling = _shard_sibling_incomplete(path)
            file_info["incomplete_sibling"] = str(sibling) if sibling else None
            if sibling:
                incomplete_shards.append(shard_name)
            else:
                missing_shards.append(shard_name)
            result["files"][shard_name] = file_info
            continue
        file_info["size"] = path.stat().st_size
        if validate_headers:
            try:
                headers = read_safetensors_header(path)
                complete_keys.update(headers)
                tensor_headers.update(headers)
                file_info["header_tensors"] = len(headers)
            except (OSError, MetadataError) as exc:
                file_info["header_error"] = str(exc)
                incomplete_shards.append(shard_name)
        result["files"][shard_name] = file_info

    result["tp_shard_plan"] = tp_shard_plan(
        tensor_headers, weight_map, tp_size
    )
    if result["tp_shard_plan"]["nondivisible_uniform_count"]:
        errors.append(
            "some converter-uniform tensors are not divisible by tp_size: "
            f"{result['tp_shard_plan']['nondivisible_uniform_count']}"
        )
    missing_keys = sorted(expected_keys - complete_keys)
    unexpected_keys = sorted(complete_keys - expected_keys)
    result["counts"].update(
        {
            "complete_tensors": len(complete_keys),
            "missing_tensors": len(missing_keys),
            "unexpected_tensors": len(unexpected_keys),
            "missing_shards": len(missing_shards),
            "incomplete_shards": len(incomplete_shards),
        }
    )
    result["missing_shards"] = missing_shards
    result["incomplete_shards"] = incomplete_shards
    if unexpected_keys:
        errors.append(f"{len(unexpected_keys)} payload tensors are not in weight_map")
    if missing_shards or incomplete_shards:
        warnings.append(
            f"checkpoint download incomplete: {len(missing_shards)} missing, "
            f"{len(incomplete_shards)} incomplete/header-invalid shards"
        )
    if missing_keys and not (missing_shards or incomplete_shards):
        errors.append(f"{len(missing_keys)} indexed tensors missing from valid headers")

    groups = _key_groups(expected_keys)
    result["groups"] = groups
    result["counts"]["engram_tensors"] = sum(
        len(names) for names in groups["engram"].values()
    )
    result["counts"]["mtp_tensors"] = sum(groups["mtp"].values())
    result["counts"]["vision_tensors"] = groups["vision_count"]
    required_engram_names = {
        "embed.weight",
        "embed.scale",
        "q_weight",
        "k_weight",
        "wkv.weight",
        "wkv.scale",
    }
    engram_complete = bool(meta and set(groups["engram"]) == set(meta.engram_layer_ids)) and all(
        set(names) == required_engram_names for names in groups["engram"].values()
    )
    engram_expected_keys = {
        key for key in expected_keys if _ENGRAM_RE.search(key)
    }
    engram_missing_keys = sorted(engram_expected_keys - complete_keys)
    result["engram_complete_manifest"] = engram_complete
    result["engram_payload_complete"] = engram_complete and not engram_missing_keys
    result["counts"]["engram_missing_tensors"] = len(engram_missing_keys)
    if engram_missing_keys:
        result["engram_missing_tensors"] = engram_missing_keys
    if meta:
        expected_engram_layers = set(meta.engram_layer_ids)
        actual_engram_layers = {int(layer) for layer in groups["engram"]}
        if expected_engram_layers != actual_engram_layers:
            errors.append(
                "Engram layer manifest mismatch: "
                f"config={sorted(expected_engram_layers)}, "
                f"index={sorted(actual_engram_layers)}"
            )
        for layer, names in groups["engram"].items():
            required = {
                "embed.weight",
                "embed.scale",
                "q_weight",
                "k_weight",
                "wkv.weight",
                "wkv.scale",
            }
            if set(names) != required:
                errors.append(f"Engram layer {layer} keys mismatch: {names}")

    expert_ids: dict[str, int] = {}
    for key in expected_keys:
        match = _EXPERT_RE.search(key)
        if match:
            prefix = "mtp" if _MTP_RE.search(key) else "backbone"
            expert_ids[prefix] = max(expert_ids.get(prefix, -1), int(match.group(1)))
    result["counts"]["backbone_experts_from_index"] = expert_ids.get("backbone", -1) + 1
    result["counts"]["mtp_experts_from_index"] = expert_ids.get("mtp", -1) + 1
    if meta and expert_ids.get("backbone", -1) + 1 not in (0, meta.n_routed_experts):
        errors.append(
            "index expert count does not match config: "
            f"index={expert_ids.get('backbone', -1) + 1}, config={meta.n_routed_experts}"
        )

    # Include shape/offset metadata for a small set of probes.  This remains
    # useful while late shards are absent: those probes are simply omitted.
    probe_suffixes = (
        "embed_tokens.weight",
        "layers.0.ffn.experts.0.w1.weight",
        "layers.0.ffn.experts.0.w2.weight",
        "layers.1.engram.embed.weight",
        "layers.14.engram.embed.weight",
    )
    for suffix in probe_suffixes:
        for key, header in tensor_headers.items():
            if key == suffix or key.endswith("." + suffix):
                result["representative_tensors"][suffix] = {
                    "key": key,
                    "shard": weight_map.get(key),
                    "dtype": header.dtype,
                    "shape": list(header.shape),
                    "row_bytes": header.row_bytes,
                }
                break
    if meta:
        engram_shards = {}
        for layer in meta.engram_layer_ids:
            rows = meta.engram_num_embeddings[meta.engram_layer_ids.index(layer)]
            shard_rows = (rows + tp_size - 1) // tp_size if tp_size else rows
            engram_shards[str(layer)] = {
                "global_rows": rows,
                "rows_per_tp_rank_with_padding": shard_rows,
                "rank_ranges": [
                    [min(rank * shard_rows, rows), min((rank + 1) * shard_rows, rows)]
                    for rank in range(max(tp_size, 1))
                ],
            }
        result["engram_shards"] = engram_shards

    if errors:
        result["status"] = "invalid"
    elif missing_shards or incomplete_shards or missing_keys:
        result["status"] = "incomplete"
    else:
        result["status"] = "ready"
    return result
