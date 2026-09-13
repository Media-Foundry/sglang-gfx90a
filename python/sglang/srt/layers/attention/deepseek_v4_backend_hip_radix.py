from __future__ import annotations

import enum
import functools
import logging
import os
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
    init_compression_metadata as _init_compression_metadata_triton,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsv4.compressor_v2 import (
    CompressorBackendMixin,
    FusedCompressMetadata,
    create_paged_compressor_data,
)
from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
    _rope_fq4,
    select_candidate_blocks,
    stable_index_topk,
    token_req_indices,
)
from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
from sglang.srt.layers.attention.dsv4.metadata import (
    PagedIndexerMetadata,
    copy_metadata,
    maybe_copy_inplace,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import (
    get_parallel,
    get_spec,
)
from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc
from sglang.srt.speculative.ragged_verify import resolve_ragged_verify_layout
from sglang.srt.utils import ceil_align

if TYPE_CHECKING:
    from sgl_kernel.flash_mla import FlashMLASchedMeta

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

SWA_WINDOW = 128
C4_TOPK = 512
PAGE_INDEX_ALIGNED_SIZE = 64


T = TypeVar("T", bound=Optional[torch.Tensor])


def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED_SIZE) -> T:
    if x is None:
        return None
    curr_size = x.shape[-1]
    target_size = ceil_align(curr_size, multiples_of)
    return F.pad(x, pad=(0, target_size - curr_size), mode="constant", value=-1)


def _create_flashmla_metadata():
    from sglang.srt.utils import is_hip

    if is_hip():
        return None
    import sgl_kernel.flash_mla as flash_mla

    return flash_mla.get_mla_metadata()[0]


def _create_dummy_paged_compress_data(compress_ratio: int):
    return None


def _low_ratio_compression_metadata(
    compress_ratio: int,
    seq_lens_casual: torch.Tensor,
    raw_out_loc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the cache locations and visible lengths for V4.1 c1/c2 pools.

    A ratio-r latent is committed only when the current causal position closes a
    complete group.  The pool is addressed in compressed-slot coordinates, so a
    full-token location is divided by ``compress_ratio``.  This is deliberately
    kept as a small tensor expression (rather than a host-side loop) because the
    same metadata is used by eager target-verify forwards.
    """
    num_write_tokens = raw_out_loc.shape[0]
    completes_group = (
        seq_lens_casual[:num_write_tokens] % compress_ratio == 0
    )
    out_loc = torch.where(
        completes_group,
        raw_out_loc.to(torch.int64) // compress_ratio,
        torch.full_like(raw_out_loc, -1, dtype=torch.int64),
    )
    visible = (seq_lens_casual // compress_ratio).clamp_min(1)
    return out_loc, visible.to(torch.int32)


def _low_ratio_sparse_buffers(
    topk_lengths_clamp1: torch.Tensor,
    topk: int,
    is_prefill: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Allocate fixed-shape sparse attention indices for one low ratio."""
    sparse_lengths = torch.clamp(topk_lengths_clamp1, max=topk)
    page_indices = _pad_last_dim(
        torch.full(
            (topk_lengths_clamp1.shape[0], topk),
            -1,
            dtype=torch.int32,
            device=topk_lengths_clamp1.device,
        )
    )
    raw_indices = torch.empty_like(page_indices) if is_prefill else None
    return sparse_lengths, page_indices, raw_indices


def _expand_low_ratio_page_table(
    page_table: torch.Tensor,
    *,
    full_page_size: int,
    compress_ratio: int,
    index_page_size: int,
) -> torch.Tensor:
    """Expand a full-token page table to the 64-slot low-ratio index pages.

    The eager HIP path currently uses the torch indexer and therefore does not
    need this table, but keeping the conversion here makes the metadata contract
    explicit and provides a safe basis for a later graph kernel.
    """
    slots_per_full_page = full_page_size // compress_ratio
    if slots_per_full_page % index_page_size:
        raise ValueError(
            f"{full_page_size=}/{compress_ratio=} is not divisible by "
            f"{index_page_size=}"
        )
    blocks_per_page = slots_per_full_page // index_page_size
    if blocks_per_page == 1:
        return page_table
    offsets = torch.arange(
        blocks_per_page, device=page_table.device, dtype=torch.int64
    )
    expanded = page_table.to(torch.int64).unsqueeze(-1) * blocks_per_page
    expanded = expanded + offsets
    return expanded.reshape(page_table.shape[0], -1).to(torch.int32)


@dataclass
class UnifiedKvMetadata:
    """
    unified-kv per-forward metadata
    """

    # SWA ring write target (req_slot*ring + pos%ring)
    swa_loc: Optional[torch.Tensor] = None

    # ragged decode index streams
    swa_indices: Optional[torch.Tensor] = None
    swa_indptr: Optional[torch.Tensor] = None
    hca_indices: Optional[torch.Tensor] = None
    hca_indptr: Optional[torch.Tensor] = None
    csa_indices: Optional[torch.Tensor] = None
    csa_indptr: Optional[torch.Tensor] = None

    # prefill/extend per-token mapping
    pf_state_slot: Optional[torch.Tensor] = None
    pf_chunk_start: Optional[torch.Tensor] = None
    pf_cu_q: Optional[torch.Tensor] = None
    pf_final_pos: Optional[torch.Tensor] = None

    # Per-token req-slot map used by the SWA ring store, precomputed once per
    # step so the forward store does not recompute a repeat_interleave per layer.
    # Read by the target-verify store (num_draft*bs tokens); for plain decode it
    # equals req_pool_indices and is unused (the decode store reads that live).
    verify_store_state_slot: Optional[torch.Tensor] = None

    # SWA-page-offset compressed-store locations (= c*_out_loc + unified_swa_pages),
    # precomputed once per step to drop the per-layer int add in the store path.
    c4_out_loc: Optional[torch.Tensor] = None
    c128_out_loc: Optional[torch.Tensor] = None

    def copy_(self, other: UnifiedKvMetadata) -> None:
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=[],
            copy_fields=[
                "swa_indices",
                "swa_indptr",
                "hca_indices",
                "hca_indptr",
                "csa_indices",
                "csa_indptr",
                "pf_state_slot",
                "pf_chunk_start",
                "pf_cu_q",
                "pf_final_pos",
                "verify_store_state_slot",
                "c4_out_loc",
                "c128_out_loc",
            ],
            # swa_loc is recomputed each forward (recorded inside cuda graphs),
            # so it is rebound rather than copied across replays.
            assign_fields=["swa_loc"],
        )


@dataclass
class DSV4AttnMetadata:
    page_size: int
    page_table: torch.Tensor
    raw_out_loc: torch.Tensor
    cuda_int32_kwargs: dict

    seq_lens_casual: torch.Tensor
    positions_casual: torch.Tensor

    swa_page_indices: torch.Tensor
    swa_topk_lengths: torch.Tensor

    c4_sparse_topk: int
    # Ratios actually allocated by the model's KV pool.  DeepSeek V4 uses
    # (4,128), while V4.1 on the HIP bring-up path uses (1,2).  Keeping this
    # explicit prevents low-ratio-only metadata from trying to manufacture a
    # nonexistent c4/c128 page table.
    present_ratios: Tuple[int, ...] = (4, 128)
    low_ratios: Tuple[int, ...] = ()
    # SWA KV-store write target (out_cache_loc translated to SWA space), computed
    # once per iteration in make_core_attn_metadata and read by the store path.
    swa_out_cache_loc: Optional[torch.Tensor] = None
    c4_out_loc: Optional[torch.Tensor] = None
    c4_topk_lengths_raw: Optional[torch.Tensor] = None
    c4_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c4_sparse_topk_lengths: torch.Tensor = field(init=False)
    c4_sparse_topk_lengths_raw: torch.Tensor = field(init=False)
    c4_sparse_page_indices: torch.Tensor = field(init=False)
    c4_sparse_raw_indices: Optional[torch.Tensor] = field(init=False, default=None)

    c128_out_loc: Optional[torch.Tensor] = None
    c128_page_indices: Optional[torch.Tensor] = None
    c128_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c128_topk_lengths_raw: Optional[torch.Tensor] = None

    # V4.1 ratio-1/2 compressed-cache locations and sparse index buffers.
    c1_out_loc: Optional[torch.Tensor] = None
    c1_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c1_sparse_topk_lengths: Optional[torch.Tensor] = field(
        init=False, default=None
    )
    c1_sparse_page_indices: Optional[torch.Tensor] = field(
        init=False, default=None
    )
    c1_sparse_raw_indices: Optional[torch.Tensor] = field(
        init=False, default=None
    )
    c2_out_loc: Optional[torch.Tensor] = None
    c2_topk_lengths_clamp1: Optional[torch.Tensor] = None
    c2_sparse_topk_lengths: Optional[torch.Tensor] = field(
        init=False, default=None
    )
    c2_sparse_page_indices: Optional[torch.Tensor] = field(
        init=False, default=None
    )
    c2_sparse_raw_indices: Optional[torch.Tensor] = field(
        init=False, default=None
    )

    # V4.1 two-level indexer state belongs to this forward, not to a layer or
    # a long-lived request slot. Store one bool per block, not per position.
    candidate_blocks: Dict[int, torch.Tensor] = field(default_factory=dict, repr=False)
    candidate_ratio: Optional[int] = None

    # unified-kv metadata
    unified: Optional[UnifiedKvMetadata] = None

    # FlashMLA scheduling metadata is kept separate for the uncompressed
    # stream (c0) and V4.1's low-ratio streams (c1/c2).  HIP/Triton currently
    # ignores the scheduler object, but keeping the distinction avoids sending
    # a c0 object to a ratio-1 call when CUDA graph support is added later.
    c0_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c1_flashmla_metadata: Optional[FlashMLASchedMeta] = field(
        init=False, default=None, repr=False
    )
    c2_flashmla_metadata: Optional[FlashMLASchedMeta] = field(
        init=False, default=None, repr=False
    )
    c4_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)
    c128_flashmla_metadata: FlashMLASchedMeta = field(init=False, repr=False)

    @property
    def positions(self) -> torch.Tensor:
        return self.positions_casual

    def get_flashmla_metadata(self, compress_ratio: Literal[0, 1, 2, 4, 128]):
        if compress_ratio == 0:
            return self.c0_flashmla_metadata
        elif compress_ratio == 1:
            return self.c1_flashmla_metadata
        elif compress_ratio == 2:
            return self.c2_flashmla_metadata
        elif compress_ratio == 4:
            return self.c4_flashmla_metadata
        elif compress_ratio == 128:
            return self.c128_flashmla_metadata
        else:
            raise ValueError(f"invalid {compress_ratio=}")

    def sparse_page_indices(self, compress_ratio: Literal[1, 2, 4]):
        if compress_ratio == 1:
            return self.c1_sparse_page_indices
        if compress_ratio == 2:
            return self.c2_sparse_page_indices
        if compress_ratio == 4:
            return self.c4_sparse_page_indices
        raise ValueError(f"invalid {compress_ratio=}")

    def sparse_raw_indices(self, compress_ratio: Literal[1, 2, 4]):
        if compress_ratio == 1:
            return self.c1_sparse_raw_indices
        if compress_ratio == 2:
            return self.c2_sparse_raw_indices
        if compress_ratio == 4:
            return self.c4_sparse_raw_indices
        raise ValueError(f"invalid {compress_ratio=}")

    def sparse_topk_lengths(self, compress_ratio: Literal[1, 2, 4]):
        if compress_ratio == 1:
            return self.c1_sparse_topk_lengths
        if compress_ratio == 2:
            return self.c2_sparse_topk_lengths
        if compress_ratio == 4:
            return self.c4_sparse_topk_lengths
        raise ValueError(f"invalid {compress_ratio=}")

    def copy_(self, other: DSV4AttnMetadata) -> None:
        copy_metadata(
            src=other,
            dst=self,
            check_eq_fields=[
                "c4_sparse_topk",
                "page_size",
                "cuda_int32_kwargs",
                "present_ratios",
                "low_ratios",
            ],
            copy_fields=[
                "raw_out_loc",
                "seq_lens_casual",
                "positions_casual",
                "c4_out_loc",
                "c128_out_loc",
                "page_table",
                "swa_page_indices",
                "swa_topk_lengths",
                "c128_page_indices",
                "c128_topk_lengths_clamp1",
                "c128_topk_lengths_raw",
                "c4_topk_lengths_raw",
                "c4_topk_lengths_clamp1",
                "c4_sparse_topk_lengths",
                "c4_sparse_topk_lengths_raw",
                "c4_sparse_page_indices",
                "c4_sparse_raw_indices",
                "c1_out_loc",
                "c1_topk_lengths_clamp1",
                "c1_sparse_topk_lengths",
                "c1_sparse_page_indices",
                "c1_sparse_raw_indices",
                "c2_out_loc",
                "c2_topk_lengths_clamp1",
                "c2_sparse_topk_lengths",
                "c2_sparse_page_indices",
                "c2_sparse_raw_indices",
                "unified",
            ],
            assign_fields=[
                # Recomputed by the recorded init_forward_metadata_in_graph op
                # each forward; not copied across replays.
                "swa_out_cache_loc",
                "candidate_blocks",
                "candidate_ratio",
                "c0_flashmla_metadata",
                "c1_flashmla_metadata",
                "c2_flashmla_metadata",
                "c4_flashmla_metadata",
                "c128_flashmla_metadata",
            ],
        )
        # Eager-only row ownership must never survive a copied/reused forward.
        # Rebind rather than clear: assign_fields temporarily aliases src.
        self.candidate_blocks = {}
        self.candidate_ratio = None

    def init_compression_metadata(self, unified_swa_pages: int = 0):
        assert self.page_table.dim() == 2
        assert (
            self.raw_out_loc.shape == self.seq_lens_casual.shape
        ), f"{self.raw_out_loc.shape=}, {self.seq_lens_casual.shape=}"

        # The metadata kernel is the legacy V4 (c4/c128) contract.  V4.1's
        # ratio-1/2 pools use a different physical page size and must not pass
        # through that kernel; doing so either allocates a bogus c4 table or
        # fails before the first request.  Keep all absent fields explicitly
        # None so callers can select the active ratio without stale state.
        self.c4_out_loc = None
        self.c4_topk_lengths_raw = None
        self.c4_topk_lengths_clamp1 = None
        self.c128_out_loc = None
        self.c128_page_indices = None
        self.c128_topk_lengths_raw = None
        self.c128_topk_lengths_clamp1 = None
        if 4 in self.present_ratios or 128 in self.present_ratios:
            (
                c4_out_loc,
                _,
                c4_topk_lengths_raw,
                c4_topk_lengths_clamp1,
                c128_out_loc,
                _,
                c128_topk_lengths_raw,
                c128_topk_lengths_clamp1,
                c128_page_indices,
            ) = _init_compression_metadata_triton(
                self.seq_lens_casual,
                self.positions_casual,
                self.raw_out_loc,
                self.page_table,
                self.page_size,
                compute_page_indices=128 in self.present_ratios,
            )
            if 4 in self.present_ratios:
                self.c4_out_loc = c4_out_loc
                self.c4_topk_lengths_raw = c4_topk_lengths_raw
                self.c4_topk_lengths_clamp1 = c4_topk_lengths_clamp1
            if 128 in self.present_ratios:
                self.c128_out_loc = c128_out_loc
                self.c128_topk_lengths_raw = c128_topk_lengths_raw
                self.c128_topk_lengths_clamp1 = c128_topk_lengths_clamp1
                self.c128_page_indices = _pad_last_dim(c128_page_indices)

        self.c1_out_loc = None
        self.c1_topk_lengths_clamp1 = None
        self.c2_out_loc = None
        self.c2_topk_lengths_clamp1 = None
        if 1 in self.low_ratios:
            self.c1_out_loc, self.c1_topk_lengths_clamp1 = (
                _low_ratio_compression_metadata(
                    1, self.seq_lens_casual, self.raw_out_loc
                )
            )
        if 2 in self.low_ratios:
            self.c2_out_loc, self.c2_topk_lengths_clamp1 = (
                _low_ratio_compression_metadata(
                    2, self.seq_lens_casual, self.raw_out_loc
                )
            )

        self.swa_page_indices = _pad_last_dim(self.swa_page_indices)

        if unified_swa_pages:
            if self.unified is None:
                self.unified = UnifiedKvMetadata()
            if self.c4_out_loc is not None:
                self.unified.c4_out_loc = self.c4_out_loc + unified_swa_pages
            if self.c128_out_loc is not None:
                self.unified.c128_out_loc = self.c128_out_loc + unified_swa_pages

    _CP_REINDEX_FIELDS = [
        "seq_lens_casual",
        "positions_casual",
        "swa_page_indices",
        "swa_topk_lengths",
        "page_table",
        "c4_topk_lengths_raw",
        "c4_topk_lengths_clamp1",
        "c128_page_indices",
        "c128_topk_lengths_clamp1",
        "c128_topk_lengths_raw",
    ]
    _CP_GLOBAL_FIELDS = [
        "raw_out_loc",
        "swa_out_cache_loc",
        "c4_out_loc",
        "c128_out_loc",
    ]

    def apply_cp_reindex(self) -> None:
        cp_rank = get_parallel().attn_cp_rank
        cp_size = get_parallel().attn_cp_size
        idx = slice(cp_rank, None, cp_size)
        pre_global_len = self.seq_lens_casual.shape[0]
        assert pre_global_len % cp_size == 0, (
            f"apply_cp_reindex: global token count {pre_global_len} is not divisible by cp_size={cp_size}. "
            "CP round-robin requires padding to ensure divisibility."
        )
        expected_local_len = pre_global_len // cp_size
        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name, None)
            assert isinstance(
                val, torch.Tensor
            ), f"CP reindex: {field_name} is {type(val)}, expected Tensor"
            setattr(self, field_name, val[idx].contiguous())

        for field_name in self._CP_REINDEX_FIELDS:
            val = getattr(self, field_name)
            assert val.shape[0] == expected_local_len, (
                f"apply_cp_reindex post-condition: {field_name}.shape[0]={val.shape[0]} "
                f"!= expected_local_len={expected_local_len} (cp_size={cp_size})"
            )
        for field_name in self._CP_GLOBAL_FIELDS:
            val = getattr(self, field_name, None)
            if val is None:
                continue
            assert val.shape[0] == pre_global_len, (
                f"apply_cp_reindex post-condition: global field {field_name}.shape[0]={val.shape[0]} "
                f"!= pre_global_len={pre_global_len} (must remain global for compressor write path)"
            )

    def init_flashmla_related(self, is_prefill: bool = False):
        # c4_sparse_topk is set from model_config.index_topk per-model
        # (small model: 512, large model: 1024).
        assert self.c4_sparse_topk in (512, 1024), (
            f"unexpected c4_sparse_topk={self.c4_sparse_topk}; "
            "supported: 512 (small) or 1024 (large)"
        )
        has_c4 = 4 in self.present_ratios
        has_c128 = 128 in self.present_ratios
        if has_c4:
            assert self.c4_topk_lengths_clamp1 is not None
            assert self.c4_topk_lengths_raw is not None
            self.c4_sparse_topk_lengths = torch.clamp(
                self.c4_topk_lengths_clamp1, max=self.c4_sparse_topk
            )
            self.c4_sparse_topk_lengths_raw = torch.clamp(
                self.c4_topk_lengths_raw, max=self.c4_sparse_topk
            )
            self.c4_sparse_page_indices = torch.full(
                (self.c4_topk_lengths_clamp1.size(0), self.c4_sparse_topk),
                -1,
                dtype=torch.int32,
                device=self.c4_topk_lengths_clamp1.device,
            )
            self.c4_sparse_page_indices = _pad_last_dim(
                self.c4_sparse_page_indices
            )
            if is_prefill:
                self.c4_sparse_raw_indices = torch.empty_like(
                    self.c4_sparse_page_indices
                )
        else:
            self.c4_sparse_topk_lengths = None
            self.c4_sparse_topk_lengths_raw = None
            self.c4_sparse_page_indices = None
            self.c4_sparse_raw_indices = None

        if 1 in self.low_ratios:
            assert self.c1_topk_lengths_clamp1 is not None
            (
                self.c1_sparse_topk_lengths,
                self.c1_sparse_page_indices,
                self.c1_sparse_raw_indices,
            ) = _low_ratio_sparse_buffers(
                self.c1_topk_lengths_clamp1, self.c4_sparse_topk, is_prefill
            )
        if 2 in self.low_ratios:
            assert self.c2_topk_lengths_clamp1 is not None
            (
                self.c2_sparse_topk_lengths,
                self.c2_sparse_page_indices,
                self.c2_sparse_raw_indices,
            ) = _low_ratio_sparse_buffers(
                self.c2_topk_lengths_clamp1, self.c4_sparse_topk, is_prefill
            )

        self.c0_flashmla_metadata = _create_flashmla_metadata()
        self.c1_flashmla_metadata = (
            _create_flashmla_metadata() if 1 in self.low_ratios else None
        )
        self.c2_flashmla_metadata = (
            _create_flashmla_metadata() if 2 in self.low_ratios else None
        )
        self.c4_flashmla_metadata = (
            _create_flashmla_metadata() if has_c4 else None
        )
        self.c128_flashmla_metadata = (
            _create_flashmla_metadata() if has_c128 else None
        )


@dataclass
class DSV4Metadata:
    core_attn_metadata: DSV4AttnMetadata
    indexer_metadata: Optional[PagedIndexerMetadata]

    # Low-ratio indexer metadata is kept separately because the existing
    # ``indexer_metadata`` name is tied to the legacy c4 page geometry.  The
    # eager HIP bring-up does not consume these objects, but retaining them
    # makes raw/graph upgrades type-safe and avoids a future c1/c2 alias bug.
    c1_indexer_metadata: Optional[PagedIndexerMetadata] = None
    c2_indexer_metadata: Optional[PagedIndexerMetadata] = None

    c4_compress_metadata: Optional[FusedCompressMetadata] = None
    c128_compress_metadata: Optional[FusedCompressMetadata] = None

    @property
    def core_metadata(self) -> DSV4AttnMetadata:
        return self.core_attn_metadata

    def copy_(self, other: DSV4Metadata):
        self.core_attn_metadata.copy_(other.core_attn_metadata)
        maybe_copy_inplace(self.indexer_metadata, src=other.indexer_metadata)
        maybe_copy_inplace(self.c1_indexer_metadata, src=other.c1_indexer_metadata)
        maybe_copy_inplace(self.c2_indexer_metadata, src=other.c2_indexer_metadata)
        maybe_copy_inplace(self.c4_compress_metadata, src=other.c4_compress_metadata)
        maybe_copy_inplace(
            self.c128_compress_metadata, src=other.c128_compress_metadata
        )


@dataclass
class DSV4RawVerifyMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    extend_seq_lens: Optional[torch.Tensor] = None

    def copy_(self, other: DSV4RawVerifyMetadata):
        self.req_pool_indices.copy_(other.req_pool_indices)
        self.seq_lens.copy_(other.seq_lens)
        self.out_cache_loc.copy_(other.out_cache_loc)

        self.extend_seq_lens = other.extend_seq_lens


@dataclass
class DSV4RawDecodeMetadata:
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    out_cache_loc: torch.Tensor

    def copy_(self, other: DSV4RawDecodeMetadata):
        self.req_pool_indices.copy_(other.req_pool_indices)
        self.seq_lens.copy_(other.seq_lens)
        self.out_cache_loc.copy_(other.out_cache_loc)


class _GraphBucket(enum.Enum):
    DECODE_OR_IDLE = "decode_or_idle"
    TARGET_VERIFY = "target_verify"
    DRAFT_EXTEND = "draft_extend"

    @classmethod
    def of(cls, forward_mode: ForwardMode) -> _GraphBucket:
        if forward_mode.is_decode_or_idle():
            return cls.DECODE_OR_IDLE
        if forward_mode.is_target_verify():
            return cls.TARGET_VERIFY
        if forward_mode.is_draft_extend_v2():
            return cls.DRAFT_EXTEND
        raise NotImplementedError(f"unsupported {forward_mode=}")


class DeepseekV4HipRadixBackend(
    AttentionBackend, C4IndexerBackendMixin, CompressorBackendMixin
):
    # DSV4 TBO runs ONLY in eager prefill (prefill cuda-graph is disabled);
    # decode/target-verify graphs are non-TBO (primary backend only). So the TBO
    # child backends must not be driven through cuda-graph capture/replay — doing
    # so rebuilds this backend's compressor/indexer metadata per replay step on
    # both children and leaks ROCm HSA resources (HSA_STATUS_ERROR_OUT_OF_RESOURCES).
    # TboAttnBackend reads this to skip children in the *_graph paths only.
    tbo_supports_cuda_graph = False
    supports_ragged_verify_graph: bool = True

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()
        self.device = torch.device(model_runner.device)
        head_dim = model_runner.model_config.head_dim
        assert (
            head_dim == 512
        ), "DSV4 MQA head_dim = qk_nope_head_dim(448) + qk_rope_head_dim(64) = 512"
        self.softmax_scale: float = head_dim**-0.5
        self.head_dim_v: int = model_runner.model_config.v_head_dim
        self.cuda_int32_kwargs = {"device": self.device, "dtype": torch.int32}
        self.swa_page_size = 128
        assert model_runner.page_size is not None
        assert model_runner.req_to_token_pool is not None
        self.page_size = model_runner.page_size
        assert self.page_size == 256, "the system hardcodes page_size=256"

        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool: DeepSeekV4TokenToKVPool = model_runner.token_to_kv_pool
        self.hisparse_coordinator = model_runner.hisparse_coordinator
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        # The V4.1 memory pool already exposes the ratios that are actually
        # allocated (1/2 for the current Flash checkpoint).  Keep this separate
        # from the per-layer ratio list so metadata creation never asks for a
        # legacy c4/c128 buffer that does not exist.
        self.present_ratios: Tuple[int, ...] = tuple(
            sorted(self.token_to_kv_pool.kv_pools)
        )
        self.low_ratios: Tuple[int, ...] = tuple(
            ratio for ratio in (1, 2) if ratio in self.present_ratios
        )
        self.has_c4 = 4 in self.present_ratios
        self.has_c128 = 128 in self.present_ratios
        self.MAX_SEQ_LEN_FOR_CAPTURE = self.req_to_token.shape[1]

        assert isinstance(self.token_to_kv_pool, DeepSeekV4TokenToKVPool)
        self.c4_topk = getattr(
            model_runner.model_config.hf_text_config, "index_topk", C4_TOPK
        )
        self.enable_deepseek_v4_fp4_indexer: bool = (
            model_runner.server_args.enable_deepseek_v4_fp4_indexer
        )
        self.topk = get_spec().speculative_eagle_topk or 0
        assert self.topk in [0, 1], "MTP Topk > 1 not supported for DeepSeek V4"
        self.mtp_enabled = self.topk > 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens: int = get_spec().speculative_num_draft_tokens
        self.is_draft_worker = getattr(model_runner, "is_draft_worker", False)
        self.is_dspark_draft = (
            self.is_draft_worker and model_runner.spec_algorithm.is_dspark()
        )
        self.is_dspark_target = (
            not getattr(model_runner, "is_draft_worker", False)
            and model_runner.spec_algorithm.is_dspark()
        )
        self.target_verify_num_draft_tokens = self.speculative_num_draft_tokens
        if self.is_dspark_draft:
            assert self.speculative_num_draft_tokens is not None
            assert self.speculative_num_draft_tokens > 1
            # DSpark draft workers verify gamma rows. The server arg keeps the
            # CUDA-side convention gamma + 1, so use an explicit effective value
            # instead of mutating speculative_num_draft_tokens in place.
            self.target_verify_num_draft_tokens = self.speculative_num_draft_tokens - 1
        self.speculative_step_id = speculative_step_id
        self.forward_metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ] = None

    def _move_to_device(self, x: List[int]) -> torch.Tensor:
        pin_tensor = torch.tensor(x, dtype=torch.int32, pin_memory=True)
        return pin_tensor.to(self.device, non_blocking=True)

    def init_forward_metadata_indexer(
        self, core_attn_metadata: DSV4AttnMetadata, compress_ratio: int = 4
    ):
        if compress_ratio == 4:
            c_seq_lens = core_attn_metadata.c4_topk_lengths_raw
            page_table = core_attn_metadata.page_table
        elif compress_ratio == 1:
            c_seq_lens = core_attn_metadata.c1_topk_lengths_clamp1
            page_table = _expand_low_ratio_page_table(
                core_attn_metadata.page_table,
                full_page_size=self.page_size,
                compress_ratio=1,
                index_page_size=self.token_to_kv_pool.get_index_k_page_size(1),
            )
        elif compress_ratio == 2:
            c_seq_lens = core_attn_metadata.c2_topk_lengths_clamp1
            page_table = _expand_low_ratio_page_table(
                core_attn_metadata.page_table,
                full_page_size=self.page_size,
                compress_ratio=2,
                index_page_size=self.token_to_kv_pool.get_index_k_page_size(2),
            )
        else:
            raise ValueError(f"unsupported indexer compress ratio {compress_ratio}")
        if c_seq_lens is None:
            return None
        return PagedIndexerMetadata(
            page_size=self.page_size,
            page_table=page_table,
            c4_seq_lens=c_seq_lens,
        )

    def init_forward_metadata_decode(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> Union[DSV4Metadata, DSV4RawDecodeMetadata]:
        assert (
            req_pool_indices.shape[0] == seq_lens.shape[0] == out_cache_loc.shape[0]
        ), f"{req_pool_indices.shape=} {seq_lens.shape=} {out_cache_loc.shape=}"

        return DSV4RawDecodeMetadata(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
        )

    def init_forward_metadata_prefill(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        out_cache_loc: torch.Tensor,
        num_tokens: int,
        extend_seq_lens: torch.Tensor,
        extend_seq_lens_cpu: List[int],
        need_compress: bool = True,
        use_prefill_cuda_graph: bool = False,
        compress_gpu_plan: bool = False,
        extend_start_loc: Optional[torch.Tensor] = None,
        attach_decode_streams: bool = False,
    ) -> DSV4Metadata:
        if extend_start_loc is not None:
            from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
                ExpandPrefillCausally,
            )

            _expanded = ExpandPrefillCausally.execute(
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                extend_seq_lens=extend_seq_lens,
                extend_start_loc=extend_start_loc,
                seq_lens_cpu=None,
                extend_seq_lens_cpu=None,
                num_tokens=num_tokens,
                padded_num_tokens=out_cache_loc.shape[0],
            )
            seq_lens_casual = _expanded.seq_lens_casual
            req_pool_indices_repeated = _expanded.req_pool_indices_repeated
        else:
            seq_lens_casual, req_pool_indices_repeated = self.expand_prefill_casually(
                num_tokens=num_tokens,
                seq_lens=seq_lens_cpu,
                extend_seq_lens=extend_seq_lens_cpu,
                req_pool_indices=req_pool_indices,
                padded_num_tokens=out_cache_loc.shape[0],
            )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            out_loc=out_cache_loc,
            need_compress=need_compress,
            is_prefill=True,
        )
        self._attach_unified_kv_prefill_meta(
            core_attn_metadata,
            req_pool_indices,
            seq_lens,
            extend_seq_lens,
            num_tokens,
            need_compress=need_compress,
        )
        if attach_decode_streams:
            # Target-verify runs through the unified_kv DECODE kernel, so build
            # per-token decode streams here. req_pool_indices_repeated is the
            # per-token (num_draft*bs -> bs) req-slot map produced by the prefill
            # expansion above.
            self._attach_unified_kv_decode_streams(
                core_attn_metadata, req_pool_indices_repeated
            )
        indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata)
            if need_compress and self.has_c4
            else None
        )
        c1_indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata, compress_ratio=1)
            if need_compress and 1 in self.low_ratios
            else None
        )
        c2_indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata, compress_ratio=2)
            if need_compress and 2 in self.low_ratios
            else None
        )
        if not need_compress:
            create = _create_dummy_paged_compress_data
        elif compress_gpu_plan:
            create = functools.partial(
                create_paged_compressor_data,
                is_prefill=True,
                token_to_kv_pool=self.token_to_kv_pool,
                req_to_token=self.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=None,
                extend_lens=extend_seq_lens,
                extend_lens_cpu=None,
                num_q_tokens=num_tokens,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
            )
        else:
            create = functools.partial(
                create_paged_compressor_data,
                is_prefill=True,
                token_to_kv_pool=self.token_to_kv_pool,
                req_to_token=self.req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                extend_lens=extend_seq_lens,
                extend_lens_cpu=extend_seq_lens_cpu,
                use_prefill_cuda_graph=use_prefill_cuda_graph,
            )
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c1_indexer_metadata=c1_indexer_metadata,
            c2_indexer_metadata=c2_indexer_metadata,
            c4_compress_metadata=(create(compress_ratio=4) if self.has_c4 else None),
            c128_compress_metadata=(
                create(compress_ratio=128) if self.has_c128 else None
            ),
        )

    def init_forward_metadata_target_verify(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: Optional[torch.Tensor] = None,
        extend_seq_lens: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
        seq_lens_cpu: Optional[List[int]] = None,
        ragged_layout=None,
    ) -> Union[DSV4Metadata, DSV4RawVerifyMetadata]:
        # HIP path: build target-verify metadata eagerly. The raw/lazy-upgrade route can
        # hit planner invariants during graph capture for DSV4+EAGLE.
        if seq_lens_cpu is None:
            seq_lens_cpu = seq_lens.tolist()
        return self.init_forward_metadata_target_verify_old(
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
            ragged_layout=ragged_layout,
        )

    def init_forward_metadata_target_verify_old(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[List[int]] = None,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
        ragged_layout=None,
    ) -> DSV4Metadata:
        batch_size = len(seq_lens)
        extend_start_loc = None
        if ragged_layout is not None:
            verify_lens_dev = ragged_layout.verify_lens.to(
                device=seq_lens.device, dtype=torch.int32
            )
            extend_start_loc = ragged_layout.extend_start_loc.to(
                device=seq_lens.device, dtype=torch.int32
            )
            extend_seq_lens = verify_lens_dev
            seq_lens = seq_lens + verify_lens_dev.to(seq_lens.dtype)
            # Total verify tokens to expand. For the graph path the padded layout
            # sets total_verify_tokens == graph_num_tokens (tier); the eager path
            # resolves a device-assembled layout whose total_verify_tokens is None,
            # so fall back to sum(verify_lens) (== real total; padded == tier).
            num_tokens = ragged_layout.total_verify_tokens
            if num_tokens is None:
                num_tokens = int(verify_lens_dev.sum().item())
            else:
                num_tokens = int(num_tokens)
            extend_seq_lens_cpu = None
            seq_lens_cpu = None
        else:
            seq_lens = seq_lens + self.target_verify_num_draft_tokens
            seq_lens_cpu = [
                x + self.target_verify_num_draft_tokens for x in seq_lens_cpu
            ]
            extend_seq_lens_cpu = [self.target_verify_num_draft_tokens] * batch_size
            num_tokens = self.target_verify_num_draft_tokens * batch_size
            extend_seq_lens = self._move_to_device(extend_seq_lens_cpu)
        if out_cache_loc is None:
            out_cache_loc = seq_lens.new_zeros(num_tokens)
        return self.init_forward_metadata_prefill(
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            num_tokens=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            need_compress=True,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
            compress_gpu_plan=ragged_layout is not None,
            extend_start_loc=extend_start_loc,
            attach_decode_streams=True,
        )

    def make_forward_metadata_from_raw_verify(
        self, raw_metadata: DSV4RawVerifyMetadata
    ) -> DSV4Metadata:
        req_pool_indices = raw_metadata.req_pool_indices
        seq_lens = raw_metadata.seq_lens
        out_cache_loc = raw_metadata.out_cache_loc

        bs, num_draft_tokens = len(seq_lens), self.target_verify_num_draft_tokens
        seq_lens = seq_lens + num_draft_tokens
        extend_seq_lens = raw_metadata.extend_seq_lens
        if extend_seq_lens is None or extend_seq_lens.numel() != bs:
            extend_seq_lens = torch.full_like(seq_lens, num_draft_tokens)
        else:
            extend_seq_lens = extend_seq_lens.to(
                device=seq_lens.device, dtype=seq_lens.dtype
            )
        extend_seq_lens = torch.minimum(extend_seq_lens, seq_lens).clamp_min_(1)

        seq_lens_casual, req_pool_indices_repeated = (
            self.expand_extend_with_same_length(
                bs, num_draft_tokens, seq_lens, req_pool_indices
            )
        )
        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
            out_loc=out_cache_loc,
            need_compress=True,
        )
        indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata)
            if self.has_c4
            else None
        )
        c1_indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata, compress_ratio=1)
            if 1 in self.low_ratios
            else None
        )
        c2_indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata, compress_ratio=2)
            if 2 in self.low_ratios
            else None
        )
        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=True,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_lens=extend_seq_lens,
            seq_lens_cpu=None,
            extend_lens_cpu=None,
            use_prefill_cuda_graph=True,
            num_q_tokens=num_draft_tokens * bs,
        )
        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c1_indexer_metadata=c1_indexer_metadata,
            c2_indexer_metadata=c2_indexer_metadata,
            c4_compress_metadata=(create(compress_ratio=4) if self.has_c4 else None),
            c128_compress_metadata=(
                create(compress_ratio=128) if self.has_c128 else None
            ),
        )

    def make_forward_metadata_from_raw_decode(
        self, raw_metadata: DSV4RawDecodeMetadata
    ) -> DSV4Metadata:
        req_pool_indices = raw_metadata.req_pool_indices
        seq_lens = raw_metadata.seq_lens
        out_cache_loc = raw_metadata.out_cache_loc

        core_attn_metadata = self.make_core_attn_metadata(
            req_to_token=self.req_to_token,
            req_pool_indices_repeated=req_pool_indices,
            seq_lens_casual=seq_lens,
            max_seq_len=self.MAX_SEQ_LEN_FOR_CAPTURE,
            out_loc=out_cache_loc,
            need_compress=True,
        )
        self._attach_unified_kv_decode_streams(core_attn_metadata, req_pool_indices)
        indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata)
            if self.has_c4
            else None
        )
        c1_indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata, compress_ratio=1)
            if 1 in self.low_ratios
            else None
        )
        c2_indexer_metadata = (
            self.init_forward_metadata_indexer(core_attn_metadata, compress_ratio=2)
            if 2 in self.low_ratios
            else None
        )

        create = functools.partial(
            create_paged_compressor_data,
            is_prefill=False,
            token_to_kv_pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
        )

        return DSV4Metadata(
            core_attn_metadata,
            indexer_metadata,
            c1_indexer_metadata=c1_indexer_metadata,
            c2_indexer_metadata=c2_indexer_metadata,
            c4_compress_metadata=(create(compress_ratio=4) if self.has_c4 else None),
            c128_compress_metadata=(
                create(compress_ratio=128) if self.has_c128 else None
            ),
        )

    def init_forward_metadata_draft_extend(
        self,
        max_seq_len: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: List[int],
        num_tokens_per_req: int,
        out_cache_loc: Optional[torch.Tensor] = None,
        use_prefill_cuda_graph: bool = False,
    ) -> DSV4Metadata:
        batch_size = len(seq_lens)
        extend_seq_lens_cpu = [num_tokens_per_req] * batch_size
        extend_seq_lens = self._move_to_device(extend_seq_lens_cpu)
        num_tokens = num_tokens_per_req * batch_size
        if out_cache_loc is None:
            out_cache_loc = seq_lens.new_zeros(num_tokens)
        return self.init_forward_metadata_prefill(
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            req_pool_indices=req_pool_indices,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
            num_tokens=num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            need_compress=False,
            use_prefill_cuda_graph=use_prefill_cuda_graph,
        )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        # Upgrade Raw->Full so the c4/c128 compress + core_attn + indexer
        # materialization is recorded inside the cuda graph; a no-op (Full
        # already) when PREP_IN_CUDA_GRAPH=0.
        if isinstance(self.forward_metadata, DSV4RawVerifyMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_verify(
                raw_metadata=self.forward_metadata,
            )
        elif isinstance(self.forward_metadata, DSV4RawDecodeMetadata):
            self.forward_metadata = self.make_forward_metadata_from_raw_decode(
                raw_metadata=self.forward_metadata,
            )

        # Compute the SWA KV-store write target once per forward and cache it on
        # the metadata for every layer's store. This is recorded inside the cuda
        # graph, so replay re-reads the live out_cache_loc buffer (spec-v2 and DP
        # padding rebind out_cache_loc after out-graph metadata prep). flash_mla
        # kernels require int32 indices.
        metadata = self.forward_metadata
        if isinstance(metadata, DSV4Metadata) and self.low_ratios:
            metadata.core_attn_metadata.candidate_blocks.clear()
            metadata.core_attn_metadata.candidate_ratio = None
        if (
            isinstance(metadata, DSV4Metadata)
            and forward_batch.out_cache_loc is not None
        ):
            out_cache_loc = forward_batch.out_cache_loc
            if (
                forward_batch.forward_mode.is_decode_or_idle()
                and self.topk > 0
                and self.speculative_num_steps > 1
            ):
                # Multi-step draft decode shares one out_cache_loc buffer across
                # steps; mirror the eager init's per-step slice.
                out_cache_loc = per_step_draft_out_cache_loc(
                    out_cache_loc,
                    forward_batch.batch_size,
                    self.topk,
                    self.speculative_num_steps,
                )[self.speculative_step_id]
            metadata.core_attn_metadata.swa_out_cache_loc = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(out_cache_loc).to(
                    torch.int32
                )
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ) -> None:
        bucket = _GraphBucket.of(forward_batch.forward_mode)
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens

        if in_capture:
            assert req_pool_indices.size(0) == bs
            assert seq_lens.size(0) == bs
            num_tokens = forward_batch.positions.numel()
            if bucket == _GraphBucket.DECODE_OR_IDLE:
                out_cache_loc = torch.zeros_like(seq_lens)
            elif bucket == _GraphBucket.TARGET_VERIFY:
                out_cache_loc = torch.zeros(num_tokens, **self.cuda_int32_kwargs)
            else:
                out_cache_loc = None
            actual_forward_mode = forward_batch.forward_mode
            seq_lens_sum = int(seq_lens.sum().item())
            seq_lens_cpu = seq_lens.cpu()
        else:
            out_cache_loc = forward_batch.out_cache_loc
            actual_forward_mode = getattr(
                forward_batch, "actual_forward_mode", forward_batch.forward_mode
            )
            seq_lens_sum = forward_batch.seq_lens_sum
            seq_lens_cpu = forward_batch.seq_lens_cpu

        if actual_forward_mode == ForwardMode.IDLE:
            logger.debug(
                f"[IDLE replay] bs={bs}, "
                f"local_seq_lens_len={len(seq_lens)}, "
                f"has_graph={bs in self.cuda_graph_metadata_of_bucket_and_bs[_GraphBucket.DECODE_OR_IDLE]}"
            )
            device = seq_lens.device
            seq_lens = torch.ones(bs, dtype=seq_lens.dtype, device=device)
            seq_lens_cpu = torch.ones(bs, dtype=torch.int64)
            seq_lens_sum = bs
            req_pool_indices = torch.zeros(
                bs, dtype=req_pool_indices.dtype, device=device
            )
            out_cache_loc = torch.zeros(bs, dtype=torch.int64, device=device)

        assert seq_lens_cpu is not None
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs]
        req_pool_indices = req_pool_indices[:bs]

        actual_max_seq_len = seq_lens_cpu.max().item()
        chosen_max_seq_len = self.MAX_SEQ_LEN_FOR_CAPTURE
        assert actual_max_seq_len <= chosen_max_seq_len

        graph_key = bs

        if bucket == _GraphBucket.DECODE_OR_IDLE:
            assert out_cache_loc is not None
            assert len(out_cache_loc.shape) == 1, f"{out_cache_loc.shape=}"
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, bs - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            temp_metadata = self.init_forward_metadata_decode(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc_padded,
            )
        elif bucket == _GraphBucket.TARGET_VERIFY:
            assert out_cache_loc is not None
            ragged_layout = resolve_ragged_verify_layout(forward_batch)
            if ragged_layout is not None:
                ragged_layout = ragged_layout.padded_to_bucket(padded_bs=bs)
                num_tokens_v = ragged_layout.graph_num_tokens
                graph_key = num_tokens_v
            else:
                num_tokens_v = self.target_verify_num_draft_tokens * bs
            out_cache_loc_padded = torch.nn.functional.pad(
                out_cache_loc,
                pad=(0, num_tokens_v - len(out_cache_loc)),
                mode="constant",
                value=0,
            )
            temp_metadata = self.init_forward_metadata_target_verify(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc_padded,
                use_prefill_cuda_graph=True,
                # CPU mirror already available here (== seq_lens, no D2H);
                # pass it so target_verify skips the per-iter seq_lens.tolist() sync.
                seq_lens_cpu=seq_lens_cpu.tolist(),
                ragged_layout=ragged_layout,
            )
        elif bucket == _GraphBucket.DRAFT_EXTEND:
            num_tokens_per_req = self.draft_extend_num_tokens_per_req
            if out_cache_loc is not None:
                # Pad the real write locations to the captured token count so
                # raw_out_loc reflects the actual replay out_cache_loc.
                out_cache_loc = torch.nn.functional.pad(
                    out_cache_loc,
                    pad=(0, num_tokens_per_req * bs - len(out_cache_loc)),
                    mode="constant",
                    value=0,
                )
            temp_metadata = self.init_forward_metadata_draft_extend(
                max_seq_len=chosen_max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                num_tokens_per_req=num_tokens_per_req,
                out_cache_loc=out_cache_loc,
                use_prefill_cuda_graph=True,
            )
        else:
            raise NotImplementedError

        self.replay_cuda_graph_metadata_from(
            bs=graph_key, temp_metadata=temp_metadata, bucket=bucket
        )

        if in_capture:
            metadata = self.forward_metadata
            self._current_capture_raw = (
                metadata
                if isinstance(
                    metadata,
                    (DSV4RawDecodeMetadata, DSV4RawVerifyMetadata),
                )
                else None
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return

        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert self.req_to_token_pool.req_to_token is self.req_to_token

        assert self.swa_page_size % SWA_WINDOW == 0 and self.page_size % 128 == 0
        assert seq_lens_cpu is not None
        max_seq_len = int(seq_lens_cpu.max().item())

        if forward_batch.forward_mode.is_decode_or_idle():
            # DSv4 bakes this step's KV write target (c4/c128) into metadata,
            # so slice the shared multi-step out_cache_loc now, not at forward time.
            out_cache_loc = forward_batch.out_cache_loc
            if self.topk > 0 and self.speculative_num_steps > 1:
                out_cache_loc = per_step_draft_out_cache_loc(
                    out_cache_loc,
                    forward_batch.batch_size,
                    self.topk,
                    self.speculative_num_steps,
                )[self.speculative_step_id]
            metadata = self.init_forward_metadata_decode(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=out_cache_loc,
            )
        elif forward_batch.forward_mode.is_target_verify():
            ragged_layout = resolve_ragged_verify_layout(forward_batch)
            metadata = self.init_forward_metadata_target_verify(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                out_cache_loc=forward_batch.out_cache_loc,
                extend_seq_lens=forward_batch.extend_seq_lens,
                seq_lens_cpu=(
                    seq_lens_cpu.tolist() if seq_lens_cpu is not None else None
                ),
                ragged_layout=ragged_layout,
            )
        elif forward_batch.forward_mode.is_prefill(include_draft_extend_v2=True):
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            extend_seq_lens = forward_batch.extend_seq_lens
            assert (
                seq_lens is not None
                and seq_lens_cpu is not None
                and extend_seq_lens is not None
                and extend_seq_lens_cpu is not None
            )
            is_draft = (
                forward_batch.forward_mode.is_draft_extend_v2() or self.is_draft_worker
            )
            metadata = self.init_forward_metadata_prefill(
                max_seq_len=max_seq_len,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu.tolist(),
                out_cache_loc=forward_batch.out_cache_loc,
                num_tokens=sum(extend_seq_lens_cpu),
                extend_seq_lens=extend_seq_lens,
                extend_seq_lens_cpu=extend_seq_lens_cpu,
                need_compress=not is_draft,
            )
        else:
            raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")

        self.forward_metadata = metadata
        self.init_forward_metadata_in_graph(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int) -> None:
        self.cuda_graph_metadata_of_bucket_and_bs: Dict[
            _GraphBucket,
            Dict[
                int,
                Union[
                    DSV4Metadata,
                    DSV4RawDecodeMetadata,
                    DSV4RawVerifyMetadata,
                ],
            ],
        ] = {bucket: {} for bucket in _GraphBucket}
        self.draft_extend_num_tokens_per_req = (
            max_num_tokens // max_bs if max_bs > 0 else 1
        )

    def replay_cuda_graph_metadata_from(
        self,
        bs: int,
        temp_metadata: Union[
            DSV4Metadata,
            DSV4RawVerifyMetadata,
            DSV4RawDecodeMetadata,
        ],
        bucket: _GraphBucket,
    ) -> None:
        if bs not in self.cuda_graph_metadata_of_bucket_and_bs[bucket]:
            # First call (from capture): store the new metadata directly.
            self.cuda_graph_metadata_of_bucket_and_bs[bucket][bs] = temp_metadata
            self.forward_metadata = temp_metadata
            return
        chosen_metadata = self.cuda_graph_metadata_of_bucket_and_bs[bucket][bs]
        chosen_metadata.copy_(temp_metadata)
        self.forward_metadata = chosen_metadata

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def on_after_cuda_graph_warmup(self):
        metadata = self.forward_metadata
        if isinstance(metadata, DSV4Metadata) and isinstance(
            metadata.core_attn_metadata, DSV4AttnMetadata
        ):
            core = metadata.core_attn_metadata
            core.c0_flashmla_metadata = _create_flashmla_metadata()
            core.c1_flashmla_metadata = (
                _create_flashmla_metadata() if 1 in core.low_ratios else None
            )
            core.c2_flashmla_metadata = (
                _create_flashmla_metadata() if 2 in core.low_ratios else None
            )
            core.c4_flashmla_metadata = (
                _create_flashmla_metadata() if 4 in core.present_ratios else None
            )
            core.c128_flashmla_metadata = (
                _create_flashmla_metadata() if 128 in core.present_ratios else None
            )

        # PREP_IN_CUDA_GRAPH=True: warmup upgraded raw->full on the host;
        # restore raw so capture re-runs the upgrade inside the graph.
        current_raw = getattr(self, "_current_capture_raw", None)
        if current_raw is not None:
            self.forward_metadata = current_raw

    # ---- DeepSeek V4.1 ratio-1/2 eager source path (HIP bring-up) ----

    def forward_low_ratio_sources(
        self,
        *,
        layer,
        x: torch.Tensor,
        q_lora: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        run_compressor: bool = True,
        run_indexer: bool = True,
    ) -> None:
        """Populate V4.1 ratio-1/2 compressed and indexer caches.

        The upstream V4.1 backend has CUDA/DeepGEMM implementations for this
        contract, but the gfx90a backend is a separate class and historically
        stopped at the old c4/c128 interface.  This first HIP implementation is
        intentionally eager and torch-based: it preserves the reference data
        flow, avoids CUDA-only kernels, and gives the attention path valid cache
        indices before any optimization is attempted.
        """
        if forward_batch.forward_mode.is_idle():
            return
        if getattr(forward_batch, "encoder_swa_replay", False):
            run_compressor = False

        n = min(x.shape[0], positions.shape[0])
        if n == 0:
            return
        x = x[:n]
        positions = positions[:n].to(torch.int64)
        if q_lora is not None:
            q_lora = q_lora[:n]
        req = token_req_indices(forward_batch, num_tokens=n)[:n]

        if run_compressor and layer.compressor is not None:
            self._low_ratio_compress_torch(layer, x, req, positions)
        if run_indexer and layer.indexer is not None:
            if q_lora is None:
                raise RuntimeError(
                    "V4.1 low-ratio indexer requires normalized q_lora"
                )
            self._low_ratio_index_topk_torch(
                layer, x, q_lora, req, positions
            )

    def _low_ratio_compress_torch(
        self,
        layer,
        x: torch.Tensor,
        req: torch.Tensor,
        pos: torch.Tensor,
    ) -> None:
        core = self.forward_metadata.core_metadata
        n = pos.shape[0]
        if n == 0:
            return
        kv, score = layer.compressor.project(x)
        ratio = layer.compress_ratio
        if ratio == 1:
            self._low_ratio_write_group(
                layer,
                kv,
                core.c1_out_loc[:n],
                pos,
            )
            return
        if ratio != 2:
            raise ValueError(f"unexpected V4.1 low ratio {ratio}")
        assert score is not None
        # ``raw_out_loc`` is a physical slot and zero is a valid allocation
        # (in particular, the first token of a fresh request).  Using
        # ``raw_out_loc == 0`` as a padding sentinel silently drops that row
        # from the ratio-2 carry state and corrupts the first pair.  The source
        # path trims graph padding before reaching here; a negative position is
        # the only unambiguous invalid-row marker.
        pad = pos < 0
        partner_kv, partner_score = self._low_ratio_pair_partners(
            layer_id=layer.layer_id,
            kv=kv,
            score=score,
            req=req,
            pos=pos,
            pad=pad,
        )
        pooled = layer.compressor.pool_pairs(
            torch.stack([partner_kv, kv], dim=1),
            torch.stack([partner_score, score], dim=1),
        )
        group_pos = torch.where(pos % 2 == 1, pos - 1, pos)
        out_loc = core.c2_out_loc[:n]
        self._low_ratio_write_group(layer, pooled, out_loc, group_pos)

    def _low_ratio_pair_partners(
        self,
        *,
        layer_id: int,
        kv: torch.Tensor,
        score: torch.Tensor,
        req: torch.Tensor,
        pos: torch.Tensor,
        pad: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read ratio-2 partners before publishing this batch into the ring."""
        state = self.token_to_kv_pool.get_attention_compress_states(layer_id)
        n = pos.shape[0]
        read_pos = (pos - 1).masked_fill(pad, -1)
        carried = state.get_state_by_state_loc(
            state.translate_from_req_position_to_state_loc(req, read_pos)
        )
        in_batch = torch.zeros_like(pad)
        if n > 1:
            in_batch[1:] = (
                (req[1:] == req[:-1])
                & (pos[1:] == pos[:-1] + 1)
                & ~pad[1:]
                & ~pad[:-1]
            )
        rolled_kv = torch.roll(kv, 1, 0)
        rolled_score = torch.roll(score, 1, 0)
        partner_kv = torch.where(in_batch[:, None], rolled_kv, carried.kv)
        partner_score = torch.where(
            in_batch[:, None], rolled_score, carried.score
        )

        # Retain the latest ring window for cross-chunk pairs.  The state API
        # maps negative positions to its sentinel row, so padded rows are safe.
        ring = state.ring_size
        keep = ~pad
        if n > ring:
            keep[:-ring] &= (req[:-ring] != req[ring:]) | pad[ring:]
        write_pos = pos.masked_fill(~keep, -1)
        state.set_state_by_state_loc(
            state.translate_from_req_position_to_state_loc(req, write_pos),
            KVAndScore.from_kv_score(kv=kv, score=score),
        )
        return partner_kv, partner_score

    def _low_ratio_write_group(
        self,
        layer,
        pooled: torch.Tensor,
        slots: torch.Tensor,
        group_pos: torch.Tensor,
    ) -> None:
        """Norm/rotate and scatter one completed c1/c2 latent group."""
        valid = slots >= 0
        if not bool(valid.any().item()):
            return
        pooled = pooled[valid]
        slots = slots[valid].to(torch.int64)
        group_pos = group_pos[valid].to(torch.int64)
        latent = layer.compressor.finish(pooled)
        freqs = layer.freqs_cis[group_pos]
        pool = self.token_to_kv_pool

        # The indexer consumes the pre-RoPE latent.  Store its packed FP4 key
        # before applying the compressed-KV RoPE/fake-quant transformation.
        if layer.indexer is not None and layer.indexer.owns_k:
            pool.set_index_k_fp4(
                layer_id=layer.layer_id,
                loc=slots,
                cache_k=layer.indexer.index_keys(latent, freqs),
            )

        latent = _rope_fq4(
            latent, freqs, layer.rope_head_dim, compressed_kv=True
        )
        pool.set_extra_key_buffer_fused(
            layer_id=layer.layer_id,
            loc=slots,
            cache_k=latent,
        )

    def _low_ratio_index_topk_torch(
        self,
        layer,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        req: torch.Tensor,
        pos: torch.Tensor,
    ) -> None:
        """Reference FP4-indexer selection for HIP eager forwards.

        Requests are processed independently so a ragged batch never compares
        scores from unrelated page tables.  The selected logical positions are
        sorted before conversion to compressed physical slots, matching the
        attention consumer's deterministic ordering contract.
        """
        core = self.forward_metadata.core_metadata
        ratio = layer.compress_ratio
        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        page_indices.fill_(-1)
        if raw_indices is not None:
            raw_indices.fill_(-1)

        indexer = layer.indexer
        if indexer.is_candidate_source:
            core.candidate_blocks.clear()
            core.candidate_ratio = ratio
        elif indexer.uses_candidates and core.candidate_ratio != ratio:
            raise RuntimeError(
                "V4.1 candidate source must run first with the same cache ratio"
            )
        q = indexer.queries(q_lora, layer.freqs_cis[pos])
        weights = indexer.head_weights(x)
        visible = (pos + 1) // ratio
        topk = indexer.index_topk

        # The scheduler lays each request's rows contiguously in the normal
        # eager path.  unique_consecutive also handles ragged verify batches.
        for request_id in torch.unique_consecutive(req).tolist():
            rows = (req == request_id).nonzero(as_tuple=False).flatten()
            if rows.numel() == 0:
                continue
            length = int(visible[rows].max().item())
            if length <= 0:
                continue
            k = min(topk, length)
            logical = torch.arange(length, device=pos.device)
            full_slots = self.req_to_token[
                int(request_id), logical * ratio
            ].to(torch.int64)
            compressed_slots = full_slots // ratio
            index_k = self.token_to_kv_pool.get_low_ratio_index_k_dequant(
                layer.layer_id, compressed_slots
            )
            scores = indexer.scores(q[rows], index_k, weights[rows])
            row_visible = visible[rows].to(torch.int64)
            scores = scores.masked_fill(
                logical[None, :] >= row_visible[:, None], -torch.inf
            )
            if indexer.is_candidate_source:
                core.candidate_blocks[request_id] = select_candidate_blocks(
                    scores,
                    row_visible[:, None],
                    indexer.candidate_topk_blocks,
                    indexer.candidate_block_size,
                )
            elif indexer.uses_candidates:
                blocks = core.candidate_blocks.get(request_id)
                expected_shape = (
                    rows.numel(),
                    (length + indexer.candidate_block_size - 1)
                    // indexer.candidate_block_size,
                )
                if blocks is None or blocks.shape != expected_shape:
                    raise RuntimeError(
                        "V4.1 missing or incompatible per-request candidate blocks"
                    )
                scores = scores.masked_fill(
                    ~blocks[:, logical // indexer.candidate_block_size], -torch.inf
                )
            selected = stable_index_topk(scores, k)
            chosen = (selected < row_visible[:, None]) & (
                scores.gather(-1, selected) > -torch.inf
            )
            selected_safe = selected.clamp_max(length - 1)
            page_indices[rows, :k] = torch.where(
                chosen,
                compressed_slots[selected_safe],
                torch.full_like(selected_safe, -1),
            ).to(torch.int32)
            if raw_indices is not None:
                raw_indices[rows, :k] = torch.where(
                    chosen,
                    selected,
                    torch.full_like(selected, -1),
                ).to(torch.int32)

    def _attach_unified_kv_decode_streams(
        self, core: DSV4AttnMetadata, state_slot: torch.Tensor
    ) -> None:
        """build the ragged decode index streams once per forward.

        ``state_slot`` is the per-row req-slot map: decode passes
        ``req_pool_indices`` (1 token per req), target-verify passes
        ``req_pool_indices_repeated`` (the per-token num_draft*bs -> bs map) so
        the same builder produces per-draft-token decode streams."""
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if not is_unified_kv_triton():
            return
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime

        pool = self.token_to_kv_pool
        N = core.positions_casual.shape[0]
        state_slot = state_slot[:N]
        if core.unified is None:
            core.unified = UnifiedKvMetadata()
        if getattr(self, "_dspark_full_block_attention", False):
            block = int(self.target_verify_num_draft_tokens)
            if block <= 0 or N % block:
                raise RuntimeError(
                    f"DSpark full-block attention needs N divisible by block: "
                    f"N={N}, block={block}"
                )
            block_end = (
                core.positions_casual.view(-1, block)[:, -1]
                .repeat_interleave(block)
                .contiguous()
            )
            visible_win = pool.unified_swa_window + block
            full_len = torch.minimum(
                block_end.to(torch.int32) + 1,
                torch.full_like(block_end, visible_win, dtype=torch.int32),
            )
            (
                core.unified.swa_indices,
                core.unified.swa_indptr,
            ) = runtime.build_swa_decode_stream(
                state_slot=state_slot,
                positions=block_end,
                swa_len=full_len,
                win=visible_win,
                ring_stride=pool.unified_swa_ring_size,
            )
        else:
            (
                core.unified.swa_indices,
                core.unified.swa_indptr,
            ) = runtime.build_swa_decode_stream(
                state_slot=state_slot,
                positions=core.positions_casual,
                swa_len=core.swa_topk_lengths,
                win=pool.unified_swa_window,
                ring_stride=pool.unified_swa_ring_size,
            )
        (
            _,
            _,
            core.unified.hca_indices,
            core.unified.hca_indptr,
            core.unified.csa_indices,
            core.unified.csa_indptr,
        ) = runtime.build_decode_streams(
            state_slot=state_slot,
            positions=core.positions_casual,
            swa_len=core.swa_topk_lengths,
            hca_len=core.c128_topk_lengths_raw,
            csa_len=core.c4_sparse_topk_lengths_raw,
            hca_page_indices=core.c128_page_indices,
            csa_width=core.c4_sparse_page_indices.shape[1],
            win=pool.unified_swa_window,
            ring_stride=pool.unified_swa_ring_size,
            swa_pages=pool.unified_swa_pages,
        )
        # SWA ring write target, same value for every layer this forward.
        req_slot = state_slot.to(torch.int64)
        core.unified.swa_loc = (
            req_slot * pool.unified_swa_ring_size
            + core.positions_casual.to(torch.int64) % pool.unified_swa_ring_size
        ).to(torch.int32)
        # Per-token req-slot map for the SWA ring store, read directly by the
        # forward store (target-verify) instead of recomputing a repeat_interleave
        # per layer. Harmless for plain decode (its store reads req_pool_indices).
        core.unified.verify_store_state_slot = state_slot

    def _attach_unified_kv_prefill_meta(
        self,
        core: DSV4AttnMetadata,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        num_tokens: int,
        need_compress: bool = True,
    ) -> None:
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if not is_unified_kv_triton():
            return
        device = req_pool_indices.device
        bs = req_pool_indices.shape[0]
        seq_lens = seq_lens.to(torch.int64)
        extend_seq_lens = extend_seq_lens.to(torch.int64)
        # token -> req index (length L = sum(extend_seq_lens)).
        # output_size skips the implicit sum() D2H on draft-extend. dropping it on the
        # target-extend path triggers a GPU memory access fault.
        if need_compress:
            bid = torch.repeat_interleave(
                torch.arange(bs, device=device, dtype=torch.int64),
                extend_seq_lens,
            )
        else:
            bid = torch.repeat_interleave(
                torch.arange(bs, device=device, dtype=torch.int64),
                extend_seq_lens,
                output_size=num_tokens,
            )
        if core.unified is None:
            core.unified = UnifiedKvMetadata()
        core.unified.pf_state_slot = req_pool_indices[bid]
        core.unified.pf_chunk_start = (seq_lens - extend_seq_lens)[bid]
        cu_q_per_req = torch.cumsum(extend_seq_lens, dim=0) - extend_seq_lens
        core.unified.pf_cu_q = cu_q_per_req[bid]
        core.unified.pf_final_pos = (seq_lens - 1)[bid]

    def _forward_unified_kv(
        self,
        *,
        q: torch.Tensor,
        kv: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        compress_ratio: Literal[0, 1, 2, 4, 128],
        attn_sink: torch.Tensor,
        core_attn_metadata: DSV4AttnMetadata,
        save_kv_cache: bool = True,
        inverse_rope_freqs: Optional[torch.Tensor] = None,
        inverse_rope_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """unified_kv paged-attention path over the bf16 unified_kv"""
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime

        pool = self.token_to_kv_pool
        layer_id = layer.layer_id
        unified = pool.get_unified_kv(layer_id)
        win = pool.unified_swa_window
        ring_stride = pool.unified_swa_ring_size
        swa_pages = pool.unified_swa_pages

        if q.ndim == 4:
            q = q.squeeze(1)
        device = q.device
        positions = forward_batch.positions.to(torch.int64)
        T = q.shape[0]
        positions = positions[:T]

        c128_pi = getattr(core_attn_metadata, "c128_page_indices", None)
        c4_pi = getattr(core_attn_metadata, "c4_sparse_page_indices", None)

        # Target-verify runs through the unified_kv DECODE kernel, same path as
        # decode; its per-token decode streams were built in metadata.
        verify_as_decode = forward_batch.forward_mode.is_target_verify()
        is_decode = forward_batch.forward_mode.is_decode_or_idle() or verify_as_decode
        if is_decode:
            if verify_as_decode:
                # Per-token (num_draft*bs -> bs) req-slot map, precomputed once
                # per step in _attach_unified_kv_decode_streams. Writing every
                # draft token's K into the ring is safe: spec_extra room prevents
                # clobbering the window history same-step tokens still read.
                state_slot = core_attn_metadata.unified.verify_store_state_slot[:T]
            else:
                state_slot = forward_batch.req_pool_indices[:T]
            if save_kv_cache:
                runtime.store_swa_into_unified(
                    kv=kv,
                    state_slot=state_slot,
                    positions=positions,
                    unified_kv=unified,
                    win=win,
                    ring_stride=ring_stride,
                    final_pos=positions,
                )
            unified_metadata = core_attn_metadata.unified
            if compress_ratio == 0:
                kv_indices = unified_metadata.swa_indices
                kv_indptr = unified_metadata.swa_indptr
            elif compress_ratio == 128:
                kv_indices = unified_metadata.hca_indices
                kv_indptr = unified_metadata.hca_indptr
            elif compress_ratio == 4:
                kv_indices = unified_metadata.csa_indices
                kv_indptr = unified_metadata.csa_indptr
                runtime.fill_compress_tail(
                    indices=kv_indices,
                    indptr=kv_indptr,
                    prefix_len=core_attn_metadata.swa_topk_lengths[:T],
                    page_indices=c4_pi[:T],
                    valid_len=core_attn_metadata.c4_sparse_topk_lengths_raw[:T],
                    swa_pages=swa_pages,
                )
            else:
                raise ValueError(f"bad compress_ratio {compress_ratio}")
            output = None
            if (
                envs.SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_CK_SPARSE_DECODE.get()
                or envs.SGLANG_DSV4_GFX90A_DSPARK_TP8_M192_CK_SPARSE_DECODE.get()
            ):
                from sglang.srt.layers.attention.dsv4_dspark_tp8 import (
                    m128_ck_eligible,
                    m192_ck_eligible,
                    refined_probability_eligible,
                )
                from sglang.srt.utils.common import is_gfx90a_supported

                gfx90a = is_gfx90a_supported()
                use_m128_ck = m128_ck_eligible(
                    enabled=envs.SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_CK_SPARSE_DECODE.get(), gfx90a=gfx90a,
                    dspark=self.is_dspark_target,
                    allow_c4=envs.SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_CK_C4.get(),
                    tp_size=get_parallel().attn_tp_size, compress_ratio=compress_ratio,
                    rows=T, batch_size=forward_batch.batch_size,
                    target_verify=forward_batch.forward_mode.is_target_verify(),
                    width=getattr(forward_batch.spec_info, "num_tokens_per_req", None),
                    inverse_rope=inverse_rope_freqs is not None or inverse_rope_positions is not None)
                use_m192_ck = m192_ck_eligible(
                    enabled=envs.SGLANG_DSV4_GFX90A_DSPARK_TP8_M192_CK_SPARSE_DECODE.get(),
                    gfx90a=gfx90a, dspark=self.is_dspark_target,
                    tp_size=get_parallel().attn_tp_size, compress_ratio=compress_ratio,
                    rows=T, batch_size=forward_batch.batch_size,
                    target_verify=forward_batch.forward_mode.is_target_verify(),
                    width=getattr(forward_batch.spec_info, "num_tokens_per_req", None),
                    inverse_rope=inverse_rope_freqs is not None or inverse_rope_positions is not None)
                if use_m128_ck or use_m192_ck:
                    from sglang.kernels.ops.attention.dsv4.gfx90a_sparse_h8 import run_if_supported

                    refined = refined_probability_eligible(
                        ck_eligible=use_m128_ck, compress_ratio=compress_ratio,
                        enabled=envs.SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_CK_C4_REFINED.get(),
                    )
                    if refined:
                        from sglang.kernels.ops.debug.gfx90a_sparse_h8_refined_probability import run_if_supported
                    output = run_if_supported(q, unified, kv_indices, kv_indptr, attn_sink, self.softmax_scale)
                    logged_attr = f"_tp8_ck_h8_c{compress_ratio}_logged"
                    if output is not None and not getattr(self, logged_attr, False):
                        logger.info("DSV4 TP8 DSpark CK H8 hit rank=%s layer=%s M%s C%s", get_parallel().attn_tp_rank, layer_id, T, compress_ratio)
                        if refined:
                            logger.info("DSV4 TP8 DSpark refined C4 probability hit rank=%s layer=%s M128", get_parallel().attn_tp_rank, layer_id)
                        setattr(self, logged_attr, True)
            if output is None:
                output = runtime.decode(
                    q=q,
                    unified_kv=unified,
                    kv_indices=kv_indices,
                    kv_indptr=kv_indptr,
                    attn_sink=attn_sink,
                    softmax_scale=self.softmax_scale,
                    inverse_rope_freqs=inverse_rope_freqs,
                    inverse_rope_positions=inverse_rope_positions,
                )
            fixture_root = envs.SGLANG_DSV4_TP8_SPARSE_FIXTURE_DIR.get()
            if (fixture_root and self.is_dspark_target
                    and forward_batch.forward_mode.is_target_verify()
                    and get_parallel().attn_tp_size == 8 and T == 128
                    and get_parallel().attn_tp_rank == 0
                    and layer_id == envs.SGLANG_DSV4_TP8_SPARSE_FIXTURE_LAYER.get()
                    and not getattr(self, "_tp8_sparse_fixture_saved", False)):
                from pathlib import Path
                from sglang.kernels.ops.debug.dsv4_tp8_sparse_fixture import save_fixture

                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("TP8 sparse fixture requires decode graphs disabled")
                if inverse_rope_freqs is not None or inverse_rope_positions is not None:
                    raise RuntimeError("TP8 sparse fixture cannot omit fused inverse RoPE")
                save_fixture(
                    Path(fixture_root) / f"layer_{layer_id}_rank_0_c{compress_ratio}.pt",
                    q=q, kv=unified, indices=kv_indices, indptr=kv_indptr,
                    sink=attn_sink, output=output, scale=self.softmax_scale,
                    provenance={
                        "kind": "eager_dspark_tp8_target",
                        "layer": layer_id, "compress_ratio": compress_ratio,
                        "positions": forward_batch.positions.detach().cpu().tolist(),
                        "input_ids": forward_batch.input_ids.detach().cpu().tolist(),
                        "ck_enabled": envs.SGLANG_DSV4_GFX90A_DSPARK_TP8_M128_CK_SPARSE_DECODE.get(),
                    },
                )
                self._tp8_sparse_fixture_saved = True
                logger.info("Saved TP8 sparse attention fixture layer=%s", layer_id)
            if os.getenv("SGLANG_DSV4_CK_REPLAY_DUMP_DIR"):
                from sglang.kernels.ops.debug.dsv4_ck_replay import (
                    maybe_dump_unified_sparse,
                )

                maybe_dump_unified_sparse(
                    layer_id=layer_id,
                    tp_rank=get_parallel().attn_tp_rank,
                    compress_ratio=compress_ratio,
                    q=q,
                    unified_kv=unified,
                    kv_indices=kv_indices,
                    kv_indptr=kv_indptr,
                    attn_sink=attn_sink,
                    output=output,
                    softmax_scale=self.softmax_scale,
                    positions=positions,
                    inverse_rope_freqs=inverse_rope_freqs,
                    inverse_rope_positions=inverse_rope_positions,
                )
            return output

        # prefill / extend
        state_slot = core_attn_metadata.unified.pf_state_slot
        chunk_start = core_attn_metadata.unified.pf_chunk_start
        cu_q = core_attn_metadata.unified.pf_cu_q
        final_pos = core_attn_metadata.unified.pf_final_pos

        # DSA CP (round-robin/interleave): unified_pf_* are built over the GLOBAL
        # token layout, but under CP each rank owns only 1/cp_size of the queries
        # (q/positions are local) while kv was all-gathered to the full sequence.
        # Slice the per-query fields to this rank's tokens so their length matches
        # the local query count T; values stay global so each local query still
        # attends over the full all-gathered KV.
        from sglang.srt.layers.attention.dsa.utils import (
            is_dsa_prefill_cp_round_robin_split,
        )

        # NOTE (AMD/HIP only): this whole DSA-CP prefill handling lives in the
        # HIP backend (DeepseekV4HipRadixBackend, selected only when is_hip()).
        # The NVIDIA path uses DeepseekV4AttnBackend and never reaches here, so
        # these CP changes do not affect B200/H200 execution.
        _cp_size = get_parallel().attn_cp_size
        _cp_active = (
            _cp_size > 1
            and is_dsa_prefill_cp_round_robin_split()
            and kv.shape[0] == _cp_size * T
            and state_slot.shape[0] != T
        )
        state_slot_full = state_slot
        final_pos_full = final_pos
        positions_full = positions
        if _cp_active:
            _sl = slice(get_parallel().attn_cp_rank, None, _cp_size)
            state_slot = state_slot[_sl].contiguous()
            chunk_start = chunk_start[_sl].contiguous()
            cu_q = cu_q[_sl].contiguous()
            final_pos = final_pos[_sl].contiguous()
            # positions for the local queries are this rank's round-robin global
            # positions {r, r+cp, r+2cp, ...}; forward_batch.positions is the full
            # (padded) global layout, so slice it the same way instead of taking
            # the first T entries (which would be the wrong, sequential 0..T-1).
            positions = forward_batch.positions.to(torch.int64)[_sl].contiguous()
            # The SWA ring must hold the FULL window on EVERY rank (decode and
            # later chunks read this rank's ring). kv was all-gathered to the full
            # sequence, so write the full kv with full global positions/state_slot
            # instead of only this rank's 1/cp_size tokens.
            positions_full = forward_batch.positions.to(torch.int64)[
                : state_slot_full.shape[0]
            ].contiguous()

        kpre_i, kpre_p, kext_i, kext_p = runtime.build_prefill_indices(
            compress_ratio=compress_ratio,
            state_slot=state_slot,
            positions=positions,
            chunk_start=chunk_start,
            cu_q=cu_q,
            win=win,
            ring_stride=ring_stride,
            swa_pages=swa_pages,
            c128_page_indices=c128_pi,
            c4_sparse_page_indices=c4_pi,
        )

        if kpre_p.shape[0] < T + 1:
            pad = T + 1 - kpre_p.shape[0]
            kpre_p = torch.cat([kpre_p, kpre_p[-1:].expand(pad)])
            kext_p = torch.cat([kext_p, kext_p[-1:].expand(pad)])
        o = runtime.prefill(
            q=q,
            unified_kv=unified,
            kv_indices_prefix=kpre_i,
            kv_indptr_prefix=kpre_p,
            kv_extend=kv,
            kv_indices_extend=kext_i,
            kv_indptr_extend=kext_p,
            attn_sink=attn_sink,
            softmax_scale=self.softmax_scale,
        )

        # write this chunk's SWA K into the ring for future chunks / decode
        # only the final-window tokens per request
        if save_kv_cache:
            # Under CP, write the FULL all-gathered window so every rank's ring is
            # complete (decode / later chunks read the local ring). Without CP this
            # is just the local kv + local metadata as before.
            _ring_state_slot = state_slot_full if _cp_active else state_slot
            _ring_final_pos = final_pos_full if _cp_active else final_pos
            _ring_positions = positions_full if _cp_active else positions
            n_real = _ring_state_slot.shape[0]
            runtime.store_swa_into_unified(
                kv=kv[:n_real],
                state_slot=_ring_state_slot,
                positions=_ring_positions[:n_real],
                unified_kv=unified,
                win=win,
                ring_stride=ring_stride,
                final_pos=_ring_final_pos,
            )
        return o

    def get_swa_out_cache_loc(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Resolve the SWA KV-store write target for the current forward.

        Fast path: the per-forward value cached by init_forward_metadata_in_graph
        (recorded inside cuda graphs, so replay re-reads live buffers). Fallback:
        translate at store time, matching the pre-cache behavior, for paths that
        never run the in-graph init — eager idle (forward_idle skips attn init),
        runners that only run the out-graph prep (e.g.
        EAGLEDraftExtendCudaGraphRunner) — or whose batch was re-padded after
        init (shape mismatch). Idle always falls back: its metadata is absent or
        left over from a previous forward, and translating the zero-padded
        out_cache_loc writes to the dummy slot.
        """
        out_cache_loc = forward_batch.out_cache_loc
        core = getattr(self.forward_metadata, "core_attn_metadata", None)
        cached = core.swa_out_cache_loc if core is not None else None
        if (
            cached is not None
            and not forward_batch.forward_mode.is_idle()
            and cached.shape[0] == out_cache_loc.shape[0]
        ):
            return cached
        return self.token_to_kv_pool.translate_loc_from_full_to_swa(out_cache_loc).to(
            torch.int32
        )

    def get_unified_swa_loc(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """SWA ring write target for unified_kv, shared by all layers.

        Fast path: the per-forward value cached in _attach_unified_kv_decode_streams
        (recorded inside cuda graphs, so replay re-reads live buffers). Fallback:
        recompute at store time, matching the pre-cache per-layer behavior, for
        paths that never ran the decode-stream init (eager prefill/extend, idle,
        or a batch re-padded after init -> shape mismatch).

        Cached swa_loc is computed once from committed positions, so every draft-decode
        step would reuse the same ring slot and break the chain. Recompute from the live
        per-step positions; only the draft path is affected, the rest keeps the fast path.
        """
        positions = forward_batch.positions
        core = getattr(self.forward_metadata, "core_attn_metadata", None)
        unified = getattr(core, "unified", None) if core is not None else None
        cached = unified.swa_loc if unified is not None else None
        is_multistep_draft_decode = (
            forward_batch.forward_mode.is_decode_or_idle()
            and self.speculative_num_steps > 1
        )
        # DSpark's draft TARGET_VERIFY graph receives live request slots and
        # positions through graph-stable input buffers, but ``UnifiedKvMetadata``
        # historically rebound ``swa_loc`` at replay.  Rebinding a Python field
        # cannot change the tensor pointer already captured by the KV-store
        # kernel, so gamma>3 replay could write every draft stage through the
        # synthetic capture-time locations.  Recompute inside the draft graph;
        # target verification and native AR retain the cached fast path.
        is_dspark_draft_verify = (
            self.is_dspark_draft
            and forward_batch.forward_mode.is_target_verify()
        )
        if (
            cached is not None
            and not forward_batch.forward_mode.is_idle()
            and cached.shape[0] == positions.shape[0]
            and not is_multistep_draft_decode
            and not is_dspark_draft_verify
        ):
            result = cached
        else:
            ring = self.token_to_kv_pool.unified_swa_ring_size
            req_slot = forward_batch.req_pool_indices.to(torch.int64)
            if req_slot.shape[0] != positions.shape[0]:
                req_slot = req_slot.repeat_interleave(
                    positions.shape[0] // req_slot.shape[0]
                )
            result = (req_slot * ring + positions.to(torch.int64) % ring).to(
                torch.int32
            )
        return result

    def store_cache(
        self, layer_id: int, swa_k: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        swa_loc = self.get_swa_out_cache_loc(forward_batch)
        self.token_to_kv_pool.set_swa_key_buffer_radix_fused(
            layer_id=layer_id,
            swa_loc=swa_loc,
            cache_k=swa_k,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        compress_ratio: Literal[0, 4, 128],
        save_kv_cache: bool = True,
        attn_sink: Optional[torch.Tensor] = None,
        inverse_rope_freqs: Optional[torch.Tensor] = None,
        inverse_rope_positions: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        if self.mtp_enabled and forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)

        assert k is v, "DeepseekV4 shares k and v"
        swa_k = k

        layer_id = layer.layer_id
        metadata = self.forward_metadata
        core_attn_metadata = metadata.core_attn_metadata
        token_to_kv_pool = self.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
            is_unified_kv_triton,
        )

        if is_unified_kv_triton():
            return self._forward_unified_kv(
                q=q,
                kv=swa_k,
                layer=layer,
                forward_batch=forward_batch,
                compress_ratio=compress_ratio,
                attn_sink=attn_sink,
                core_attn_metadata=core_attn_metadata,
                save_kv_cache=save_kv_cache,
                inverse_rope_freqs=inverse_rope_freqs,
                inverse_rope_positions=inverse_rope_positions,
            )

        if isinstance(core_attn_metadata, DSV4AttnMetadata):
            if save_kv_cache:
                self.store_cache(layer_id, swa_k, forward_batch)
            swa_k_cache = token_to_kv_pool.get_swa_key_buffer_radix(layer_id)

            extra_k_cache, extra_indices, extra_topk_lengths = None, None, None
            if compress_ratio == 4:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c4_sparse_page_indices
                extra_topk_lengths = core_attn_metadata.c4_sparse_topk_lengths
            elif compress_ratio in (1, 2):
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.sparse_page_indices(compress_ratio)
                extra_topk_lengths = core_attn_metadata.sparse_topk_lengths(
                    compress_ratio
                )
            elif compress_ratio == 128:
                extra_k_cache = token_to_kv_pool.get_extra_key_buffer(layer_id)
                extra_indices = core_attn_metadata.c128_page_indices
                extra_topk_lengths = core_attn_metadata.c128_topk_lengths_clamp1

            swa_window_size = token_to_kv_pool.swa_window_size
            assert swa_k_cache.ndim == 2
            k_cache_total_dim = token_to_kv_pool.swa_kv_pool.kv_cache_total_dim
            swa_k_cache = swa_k_cache[:, : swa_window_size * k_cache_total_dim].view(
                swa_k_cache.shape[0], swa_window_size, 1, k_cache_total_dim
            )

            if extra_k_cache is not None:
                page_sizes = {
                    1: token_to_kv_pool.page_size // 1,
                    2: token_to_kv_pool.page_size // 2,
                    4: token_to_kv_pool.page_size // 4,
                    128: token_to_kv_pool.page_size // 128,
                }
                extra_k_cache = extra_k_cache[
                    :, : page_sizes[compress_ratio] * k_cache_total_dim
                ].view(
                    extra_k_cache.shape[0],
                    page_sizes[compress_ratio],
                    1,
                    k_cache_total_dim,
                )
            swa_page_indices = core_attn_metadata.swa_page_indices
            swa_topk_lengths = core_attn_metadata.swa_topk_lengths

            if self.mtp_enabled:
                if swa_page_indices.shape[0] != q.shape[0]:
                    swa_page_indices = _pad_tensor_to_size(
                        swa_page_indices, q.shape[0], value=0
                    )

                if swa_topk_lengths.shape[0] != q.shape[0]:
                    swa_topk_lengths = _pad_tensor_to_size(
                        swa_topk_lengths, q.shape[0], value=1
                    )

            if q.ndim == 3:
                q = q.unsqueeze(1)
            if swa_page_indices.ndim == 2:
                swa_page_indices = swa_page_indices.unsqueeze(1)
            if extra_indices is not None and extra_indices.ndim == 2:
                extra_indices = extra_indices.unsqueeze(1)

            assert attn_sink is not None

            flashmla_metadata = core_attn_metadata.get_flashmla_metadata(compress_ratio)

            assert (
                swa_page_indices.shape[-1] % 64 == 0
            ), f"{swa_page_indices.shape=}'s last dimension is not aligned to 64"
            if extra_indices is not None:
                assert (
                    extra_indices.shape[-1] % 64 == 0
                ), f"{extra_indices.shape=}'s last dimension is not aligned to 64"

            from sglang.srt.layers.attention.hip_flash_mla import (
                flash_mla_with_kvcache_entrypoint,
            )

            backend = envs.SGLANG_HACK_FLASHMLA_BACKEND.get()
            input_dict = dict(
                q=q,
                k_cache=swa_k_cache,
                head_dim_v=self.head_dim_v,
                block_table=None,
                cache_seqlens=None,
                tile_scheduler_metadata=flashmla_metadata,
                softmax_scale=self.softmax_scale,
                is_fp8_kvcache=True,
                indices=swa_page_indices,
                topk_length=swa_topk_lengths,
                attn_sink=attn_sink,
                extra_k_cache=extra_k_cache,
                extra_indices_in_kvcache=extra_indices,
                extra_topk_length=extra_topk_lengths,
            )
            o = flash_mla_with_kvcache_entrypoint(**input_dict, backend=backend)[0]

            o = o.squeeze(1)
            return o

        raise NotImplementedError("ragged attention")

    def expand_prefill_casually(
        self,
        num_tokens: int,
        seq_lens: List[int],
        extend_seq_lens: List[int],
        req_pool_indices: torch.Tensor,
        padded_num_tokens: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_lens_casual = torch.empty(num_tokens, **self.cuda_int32_kwargs)
        idx_to_req_repeated = torch.empty(num_tokens, **self.cuda_int32_kwargs)
        offset = 0
        for i, (kv_len, qo_len) in enumerate(zip(seq_lens, extend_seq_lens)):
            out = seq_lens_casual[offset : offset + qo_len]
            offset += qo_len
            torch.arange(kv_len - qo_len + 1, kv_len + 1, out=out)
            idx_to_req_repeated[offset - qo_len : offset].fill_(i)

        assert offset == num_tokens
        req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]

        if padded_num_tokens is not None and padded_num_tokens > num_tokens:
            pad_size = padded_num_tokens - num_tokens
            seq_lens_casual = torch.nn.functional.pad(
                seq_lens_casual,
                (0, pad_size),
                value=1,
            )
            req_pool_indices_repeated = torch.nn.functional.pad(
                req_pool_indices_repeated,
                (0, pad_size),
                value=req_pool_indices_repeated[-1].item(),
            )

        return seq_lens_casual, req_pool_indices_repeated

    def expand_extend_with_same_length(
        self,
        bs: int,
        qo_len: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
    ):
        seq_lens_casual = seq_lens[:, None] + torch.arange(
            -qo_len + 1, 1, **self.cuda_int32_kwargs
        )
        seq_lens_casual = seq_lens_casual.flatten()
        idx_to_req_repeated = torch.arange(
            bs, **self.cuda_int32_kwargs
        ).repeat_interleave(qo_len)
        req_pool_indices_repeated = req_pool_indices[idx_to_req_repeated]
        return seq_lens_casual, req_pool_indices_repeated

    def make_core_attn_metadata(
        self,
        req_to_token: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
        seq_lens_casual: torch.Tensor,
        max_seq_len: int,
        out_loc: torch.Tensor,
        need_compress: bool = True,
        is_prefill: bool = False,
    ) -> DSV4AttnMetadata:
        assert self.swa_page_size == SWA_WINDOW

        seq_lens_casual = seq_lens_casual.to(torch.int32)

        swa_page_indices = self.get_swa_page_indices(
            seq_lens_casual=seq_lens_casual,
            req_pool_indices_repeated=req_pool_indices_repeated,
        )

        swa_page_indices = _pad_last_dim(
            swa_page_indices, multiples_of=PAGE_INDEX_ALIGNED_SIZE
        )

        raw_positions = seq_lens_casual - 1
        swa_topk_lengths = torch.clamp(seq_lens_casual, max=SWA_WINDOW)

        page_table = req_to_token[
            req_pool_indices_repeated, : max_seq_len : self.page_size
        ]
        page_table = (page_table // self.page_size).to(torch.int32)

        core_attn_metadata = DSV4AttnMetadata(
            page_size=self.page_size,
            raw_out_loc=out_loc,
            seq_lens_casual=seq_lens_casual,
            cuda_int32_kwargs=self.cuda_int32_kwargs,
            positions_casual=raw_positions,
            page_table=page_table,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            c4_sparse_topk=self.c4_topk,
            present_ratios=self.present_ratios,
            low_ratios=self.low_ratios,
        )

        if need_compress:
            core_attn_metadata.init_compression_metadata(
                unified_swa_pages=getattr(self.token_to_kv_pool, "unified_swa_pages", 0)
            )
            core_attn_metadata.init_flashmla_related(is_prefill=is_prefill)
        else:
            core_attn_metadata.c4_sparse_topk_lengths = None
            core_attn_metadata.c4_sparse_topk_lengths_raw = None
            core_attn_metadata.c4_sparse_page_indices = None
            core_attn_metadata.c4_sparse_raw_indices = None
            core_attn_metadata.c0_flashmla_metadata = _create_flashmla_metadata()
            core_attn_metadata.c1_flashmla_metadata = None
            core_attn_metadata.c2_flashmla_metadata = None
            core_attn_metadata.c4_flashmla_metadata = None
            core_attn_metadata.c128_flashmla_metadata = None
        return core_attn_metadata

    def get_swa_page_indices(
        self,
        seq_lens_casual: torch.Tensor,
        req_pool_indices_repeated: torch.Tensor,
    ) -> torch.Tensor:
        pos_causal = seq_lens_casual - 1
        num_qo_tokens = seq_lens_casual.size(0)
        offsets = pos_causal.unsqueeze(1) - torch.arange(
            SWA_WINDOW, **self.cuda_int32_kwargs
        ).unsqueeze(0)
        invalid_offset_mask = offsets < 0
        offsets.masked_fill_(invalid_offset_mask, 0)
        raw_indices = self.req_to_token[req_pool_indices_repeated[:, None], offsets]
        assert raw_indices.shape == (num_qo_tokens, SWA_WINDOW)
        raw_indices.masked_fill_(invalid_offset_mask, -1)
        swa_indices = self.token_to_kv_pool.translate_loc_from_full_to_swa(raw_indices)
        # flash_mla attention requires int32 page indices.
        return swa_indices.to(torch.int32)


class DeepseekV4MultiStepBackend(DeepseekV4HipRadixBackend):
    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        super().__init__(model_runner)
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends: List[DeepseekV4HipRadixBackend] = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                DeepseekV4HipRadixBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from types import SimpleNamespace

        inner_fb = SimpleNamespace(
            batch_size=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
            # Propagate the real runtime mode so inner backends can detect IDLE
            # and apply their idle substitution.
            actual_forward_mode=getattr(
                forward_batch, "actual_forward_mode", forward_batch.forward_mode
            ),
            input_ids=getattr(forward_batch, "input_ids", None),
            positions=getattr(forward_batch, "positions", None),
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
            encoder_lens=None,
            out_cache_loc=getattr(forward_batch, "out_cache_loc", None),
            spec_info=forward_batch.spec_info,
        )
        if in_capture:
            for i in range(self.speculative_num_steps):
                self.attn_backends[i].init_forward_metadata_out_graph(
                    inner_fb, in_capture=True
                )
        else:
            if self.speculative_num_steps == 1:
                return
            self.attn_backends[0].init_forward_metadata_out_graph(inner_fb)
            temp_metadata = self.attn_backends[0].forward_metadata
            for i in range(1, self.speculative_num_steps - 1):
                self.attn_backends[i].replay_cuda_graph_metadata_from(
                    bs=forward_batch.batch_size,
                    temp_metadata=temp_metadata,
                    bucket=_GraphBucket.DECODE_OR_IDLE,
                )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def on_after_cuda_graph_warmup(self):
        for backend in self.attn_backends:
            backend.on_after_cuda_graph_warmup()


def _pad_tensor_to_size(tensor: torch.Tensor, size: int, *, value: int = 0):
    if value == 0:
        return torch.cat(
            [tensor, tensor.new_zeros(size - tensor.shape[0], *tensor.shape[1:])],
            dim=0,
        )
    else:
        return torch.cat(
            [
                tensor,
                tensor.new_full((size - tensor.shape[0], *tensor.shape[1:]), value),
            ],
            dim=0,
        )
