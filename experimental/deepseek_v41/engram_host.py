"""RAM-backed Engram table primitives for the DeepSeek-V4.1 bring-up.

The Engram tables are intentionally *not* loaded into HBM by this module.  A
table is a read-only mmap of its safetensors tensor, and only a bounded batch of
rows is copied into a small staging pool when a caller explicitly prefetches
it.  This keeps the design compatible with the 1M-token KV budget and with a
machine whose memlock limit is far below the size of an Engram table.

This is an integration boundary, not a model implementation.  It preserves
raw on-disk bytes (including FP8 encodings) so a later GPU consumer can choose
the exact decode/conversion contract after the complete checkpoint is
available.
"""

from __future__ import annotations

import json
import logging
import mmap
import re
import resource
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch

from .metadata import SafetensorsTensorHeader, read_safetensors_header

logger = logging.getLogger(__name__)

_ENGRAM_KEY_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.engram\.(.+)$")


class RowStore(Protocol):
    """Minimal byte-oriented row source used by the host table."""

    @property
    def rows(self) -> int: ...

    @property
    def row_bytes(self) -> int: ...

    @property
    def dtype(self) -> str: ...

    def read_rows_bytes(self, row_ids: Sequence[int]) -> list[bytes]: ...

    def read_all_bytes(self) -> bytes: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class EngramRowMapping:
    """Exact dim-0 row sharding used by the V4.1 converter.

    The converter uses ``ceil(global_rows / tp_size)`` and pads the final local
    shard.  ``global_start``/``global_end`` describe real rows; the padded tail
    is represented by ``local_rows_with_padding`` and never maps to a global
    row.  Keeping this distinction prevents an out-of-range hash id from being
    silently redirected to a valid table row.
    """

    global_rows: int
    tp_size: int = 1
    rank: int = 0

    def __post_init__(self) -> None:
        if self.global_rows <= 0:
            raise ValueError("global_rows must be positive")
        if self.tp_size <= 0:
            raise ValueError("tp_size must be positive")
        if self.rank < 0 or self.rank >= self.tp_size:
            raise ValueError(f"rank {self.rank} is outside tp_size {self.tp_size}")

    @property
    def local_rows_with_padding(self) -> int:
        return (self.global_rows + self.tp_size - 1) // self.tp_size

    @property
    def global_start(self) -> int:
        return min(self.rank * self.local_rows_with_padding, self.global_rows)

    @property
    def global_end(self) -> int:
        return min(
            (self.rank + 1) * self.local_rows_with_padding,
            self.global_rows,
        )

    @property
    def local_rows(self) -> int:
        return self.global_end - self.global_start

    def rank_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                min(rank * self.local_rows_with_padding, self.global_rows),
                min((rank + 1) * self.local_rows_with_padding, self.global_rows),
            )
            for rank in range(self.tp_size)
        )

    def owner_rank(self, row_id: int) -> int:
        row_id = int(row_id)
        if row_id < 0 or row_id >= self.global_rows:
            raise IndexError(
                f"global Engram row {row_id} outside [0, {self.global_rows})"
            )
        return min(row_id // self.local_rows_with_padding, self.tp_size - 1)

    def global_to_local(
        self, row_ids: Sequence[int], *, strict: bool = True
    ) -> tuple[int, ...]:
        result = []
        for row_id in row_ids:
            row_id = int(row_id)
            owned = self.global_start <= row_id < self.global_end
            if not owned:
                if strict:
                    raise IndexError(
                        f"global Engram row {row_id} is not owned by "
                        f"rank {self.rank} [{self.global_start}, {self.global_end})"
                    )
                result.append(-1)
            else:
                result.append(row_id - self.global_start)
        return tuple(result)

    def local_to_global(
        self, local_ids: Sequence[int], *, allow_padding: bool = False
    ) -> tuple[int, ...]:
        result = []
        for local_id in local_ids:
            local_id = int(local_id)
            if local_id < 0 or local_id >= self.local_rows_with_padding:
                raise IndexError(
                    f"local Engram row {local_id} is outside "
                    f"[0, {self.local_rows_with_padding})"
                )
            global_id = self.global_start + local_id
            if global_id >= self.global_rows and not allow_padding:
                raise IndexError(f"local Engram row {local_id} is padding")
            result.append(global_id)
        return tuple(result)


class _MMapRowStore:
    """Common mmap implementation for a contiguous row-major byte range."""

    def __init__(
        self,
        *,
        path: Path,
        rows: int,
        row_bytes: int,
        dtype: str,
        data_start: int,
        data_end: int,
        row_offset: int = 0,
    ) -> None:
        if rows <= 0 or row_bytes <= 0:
            raise ValueError("rows and row_bytes must be positive")
        if row_offset < 0:
            raise ValueError("row_offset must be non-negative")
        expected = rows * row_bytes
        if data_end - data_start < expected:
            raise ValueError(
                f"row range is too short: need {expected} bytes, "
                f"have {data_end - data_start}"
            )
        self.path = Path(path)
        self._rows = rows
        self._row_bytes = row_bytes
        self._dtype = dtype
        self._data_start = data_start
        self._data_end = data_end
        self._row_offset = row_offset
        self._lock = threading.Lock()
        self._file = None
        self._mmap: mmap.mmap | None = None

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def row_bytes(self) -> int:
        return self._row_bytes

    @property
    def dtype(self) -> str:
        return self._dtype

    def _ensure_mmap(self) -> mmap.mmap:
        with self._lock:
            if self._mmap is None:
                self._file = self.path.open("rb")
                self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
                # Engram probes are intentionally random; avoid kernel
                # read-ahead when the platform exposes madvise.
                if hasattr(self._mmap, "madvise") and hasattr(mmap, "MADV_RANDOM"):
                    self._mmap.madvise(mmap.MADV_RANDOM)
            return self._mmap

    def read_rows_bytes(self, row_ids: Sequence[int]) -> list[bytes]:
        if not row_ids:
            return []
        mm = self._ensure_mmap()
        result: list[bytes] = []
        with self._lock:
            for row_id in row_ids:
                row_id = int(row_id)
                if row_id < 0 or row_id >= self._rows:
                    raise IndexError(f"row {row_id} outside [0, {self._rows})")
                begin = self._data_start + (self._row_offset + row_id) * self._row_bytes
                end = begin + self._row_bytes
                if end > self._data_end:
                    raise IndexError(f"row {row_id} exceeds mapped tensor range")
                result.append(bytes(mm[begin:end]))
        return result

    def read_all_bytes(self) -> bytes:
        """Return a non-row-addressed tensor payload as raw bytes.

        This method is intentionally explicit: callers must not use it for the
        huge ``embed.weight``/``embed.scale`` tables.
        """
        mm = self._ensure_mmap()
        with self._lock:
            return bytes(mm[self._data_start : self._data_end])

    def close(self) -> None:
        with self._lock:
            if self._mmap is not None:
                self._mmap.close()
                self._mmap = None
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self) -> "_MMapRowStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class SafetensorsRowStore(_MMapRowStore):
    """Read rows directly from one contiguous safetensors tensor."""

    def __init__(
        self,
        header: SafetensorsTensorHeader,
        *,
        row_offset: int = 0,
        rows: int | None = None,
    ) -> None:
        if header.row_bytes is None or not header.shape:
            raise ValueError(
                f"{header.key}: dtype {header.dtype!r} has no known row width"
            )
        available_rows = int(header.shape[0])
        if row_offset < 0 or row_offset > available_rows:
            raise ValueError(f"row_offset {row_offset} outside tensor")
        rows = available_rows - row_offset if rows is None else int(rows)
        if rows < 0 or row_offset + rows > available_rows:
            raise ValueError("requested rows exceed safetensors tensor")
        super().__init__(
            path=header.path,
            rows=rows,
            row_bytes=header.row_bytes,
            dtype=header.dtype,
            data_start=header.data_start,
            data_end=header.data_end,
            row_offset=row_offset,
        )
        self.key = header.key
        self.shape = header.shape
        self.header = header

    @classmethod
    def from_file(cls, path: str | Path, key: str) -> "SafetensorsRowStore":
        path = Path(path)
        headers = read_safetensors_header(path)
        try:
            header = headers[key]
        except KeyError as exc:
            raise KeyError(f"{key!r} not found in {path}") from exc
        return cls(header)


class RawMMapRowStore(_MMapRowStore):
    """Optional row source for a future packed/raw Engram export."""

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        rows: int,
        row_bytes: int,
        dtype: str = "RAW",
        offset: int = 0,
    ) -> "RawMMapRowStore":
        path = Path(path)
        size = path.stat().st_size
        end = offset + rows * row_bytes
        if offset < 0 or end > size:
            raise ValueError(f"raw row range [{offset}, {end}) exceeds {path} ({size})")
        if offset % row_bytes:
            raise ValueError("raw row offset must be aligned to row_bytes")
        return cls(
            path=path,
            rows=rows,
            row_bytes=row_bytes,
            dtype=dtype,
            data_start=0,
            data_end=size,
            row_offset=offset // row_bytes,
        )


@dataclass
class EngramLayerHostTable:
    """Byte-preserving host sources for one Engram layer."""

    layer_id: int
    stores: Mapping[str, RowStore]
    mapping: EngramRowMapping
    source_is_local_shard: bool = False

    # Only these tensors are indexed by the 384M-row hash id.  The q/k/wkv
    # tensors have small operator dimensions and must never be indexed with a
    # hash row id.
    ROW_TENSORS = frozenset(("embed.weight", "embed.scale"))

    def __post_init__(self) -> None:
        if not self.stores:
            raise ValueError("an Engram layer needs at least one tensor store")
        for name, store in self.stores.items():
            if name not in self.ROW_TENSORS:
                continue
            if (
                store.rows < self.mapping.local_rows_with_padding
                and self.source_is_local_shard
            ):
                raise ValueError(
                    f"{name} has {store.rows} rows, below padded shard size "
                    f"{self.mapping.local_rows_with_padding}"
                )
        weight = self.stores.get("embed.weight")
        scale = self.stores.get("embed.scale")
        if weight is not None and scale is not None and weight.rows != scale.rows:
            raise ValueError(
                "embed.weight/embed.scale row counts differ: "
                f"{weight.rows} != {scale.rows}"
            )

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(self.stores.keys())

    @property
    def row_tensor_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.stores if name in self.ROW_TENSORS)

    @property
    def static_tensor_names(self) -> tuple[str, ...]:
        return tuple(name for name in self.stores if name not in self.ROW_TENSORS)

    def _source_ids(self, global_row_ids: Sequence[int]) -> tuple[int, ...]:
        if self.source_is_local_shard:
            return self.mapping.global_to_local(global_row_ids)
        result = tuple(int(row_id) for row_id in global_row_ids)
        for row_id in result:
            if row_id < 0 or row_id >= self.mapping.global_rows:
                raise IndexError(
                    f"global Engram row {row_id} outside [0, {self.mapping.global_rows})"
                )
        return result

    def read_rows_bytes(
        self,
        global_row_ids: Sequence[int],
        *,
        tensor_names: Sequence[str] | None = None,
    ) -> dict[str, list[bytes]]:
        source_ids = self._source_ids(global_row_ids)
        names = self.row_tensor_names if tensor_names is None else tuple(tensor_names)
        result: dict[str, list[bytes]] = {}
        for name in names:
            if name not in self.stores:
                raise KeyError(f"Engram layer {self.layer_id} has no tensor {name!r}")
            if name not in self.ROW_TENSORS:
                raise ValueError(
                    f"{name} is a static Engram tensor; use read_static_bytes()"
                )
            result[name] = self.stores[name].read_rows_bytes(source_ids)
        return result

    def read_static_bytes(
        self, tensor_names: Sequence[str] | None = None
    ) -> dict[str, bytes]:
        """Read explicitly requested non-row-addressed tensors."""
        names = self.static_tensor_names if tensor_names is None else tuple(tensor_names)
        result: dict[str, bytes] = {}
        for name in names:
            if name not in self.stores:
                raise KeyError(f"Engram layer {self.layer_id} has no tensor {name!r}")
            if name in self.ROW_TENSORS:
                raise ValueError(f"{name} is row-addressed; use read_rows_bytes()")
            result[name] = self.stores[name].read_all_bytes()
        return result

    def close(self) -> None:
        for store in self.stores.values():
            store.close()


class EngramHostTable:
    """Collection of RAM-backed Engram layers, with no implicit HBM allocation."""

    REQUIRED_TENSORS = (
        "embed.weight",
        "embed.scale",
        "q_weight",
        "k_weight",
        "wkv.weight",
        "wkv.scale",
    )

    def __init__(
        self,
        layers: Mapping[int, EngramLayerHostTable],
        *,
        missing_tensors: Sequence[str] = (),
    ) -> None:
        self.layers = dict(layers)
        self.missing_tensors = tuple(missing_tensors)

    @property
    def complete(self) -> bool:
        return not self.missing_tensors and bool(self.layers)

    def describe(self) -> dict[str, Any]:
        return {
            "layers": sorted(self.layers),
            "complete": self.complete,
            "missing_count": len(self.missing_tensors),
            "missing_preview": list(self.missing_tensors[:8]),
            "row_ranges": {
                str(layer_id): {
                    "global_rows": table.mapping.global_rows,
                    "rank": table.mapping.rank,
                    "tp_size": table.mapping.tp_size,
                    "global_start": table.mapping.global_start,
                    "global_end": table.mapping.global_end,
                }
                for layer_id, table in sorted(self.layers.items())
            },
            "tensor_kinds": {
                str(layer_id): {
                    "row": list(table.row_tensor_names),
                    "static": list(table.static_tensor_names),
                }
                for layer_id, table in sorted(self.layers.items())
            },
        }

    @classmethod
    def from_safetensors_index(
        cls,
        model_dir: str | Path,
        *,
        index_path: str | Path | None = None,
        tp_size: int = 1,
        rank: int = 0,
        allow_incomplete: bool = True,
    ) -> "EngramHostTable":
        """Build lazy stores from an index without reading tensor payloads."""

        root = Path(model_dir)
        index_file = Path(index_path) if index_path else root / "model.safetensors.index.json"
        with index_file.open() as stream:
            index = json.load(stream)
        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, Mapping):
            raise ValueError(f"{index_file}: weight_map must be an object")

        grouped: dict[int, dict[str, tuple[str, str]]] = {}
        for key, shard in weight_map.items():
            match = _ENGRAM_KEY_RE.search(str(key))
            if not match:
                continue
            layer_id = int(match.group(1))
            tensor_name = match.group(2)
            grouped.setdefault(layer_id, {})[tensor_name] = (str(key), str(shard))

        layers: dict[int, EngramLayerHostTable] = {}
        missing: list[str] = []
        # The global row counts are read from embed.weight headers, not guessed
        # from config.  This is important because the two V4.1 tables differ by
        # several thousand rows and converter padding is rank-dependent.
        for layer_id, entries in sorted(grouped.items()):
            embed_entry = entries.get("embed.weight")
            if embed_entry is None:
                missing.append(f"layers.{layer_id}.engram.embed.weight")
                continue
            embed_key, embed_shard = embed_entry
            embed_path = root / embed_shard
            if not embed_path.exists():
                missing.append(str(embed_path))
                continue
            try:
                embed_store = SafetensorsRowStore.from_file(embed_path, embed_key)
            except (OSError, ValueError, KeyError) as exc:
                missing.append(f"{embed_path}:{embed_key} ({exc})")
                continue
            mapping = EngramRowMapping(
                global_rows=embed_store.rows,
                tp_size=tp_size,
                rank=rank,
            )
            stores: dict[str, RowStore] = {"embed.weight": embed_store}
            for tensor_name, (key, shard) in entries.items():
                if tensor_name == "embed.weight":
                    continue
                path = root / shard
                if not path.exists():
                    missing.append(str(path))
                    continue
                try:
                    store = SafetensorsRowStore.from_file(path, key)
                except (OSError, ValueError, KeyError) as exc:
                    missing.append(f"{path}:{key} ({exc})")
                    continue
                stores[tensor_name] = store
            absent = set(cls.REQUIRED_TENSORS) - set(stores)
            missing.extend(
                f"layers.{layer_id}.engram.{name}" for name in sorted(absent)
            )
            layers[layer_id] = EngramLayerHostTable(
                layer_id=layer_id,
                stores=stores,
                mapping=mapping,
                source_is_local_shard=False,
            )

        if missing and not allow_incomplete:
            raise FileNotFoundError(
                "incomplete Engram checkpoint; missing/invalid entries:\n"
                + "\n".join(missing[:32])
            )
        return cls(layers, missing_tensors=missing)

    def read_rows_bytes(
        self,
        layer_id: int,
        row_ids: Sequence[int],
        *,
        tensor_names: Sequence[str] | None = None,
    ) -> dict[str, list[bytes]]:
        try:
            layer = self.layers[int(layer_id)]
        except KeyError as exc:
            raise KeyError(f"Engram layer {layer_id} is not available") from exc
        return layer.read_rows_bytes(row_ids, tensor_names=tensor_names)

    def read_static_bytes(
        self, layer_id: int, *, tensor_names: Sequence[str] | None = None
    ) -> dict[str, bytes]:
        try:
            layer = self.layers[int(layer_id)]
        except KeyError as exc:
            raise KeyError(f"Engram layer {layer_id} is not available") from exc
        return layer.read_static_bytes(tensor_names=tensor_names)

    def close(self) -> None:
        for layer in self.layers.values():
            layer.close()

    def __enter__(self) -> "EngramHostTable":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@dataclass
class StagingLease:
    """A slot in :class:`PinnedStagingPool`; release exactly once."""

    tensor: torch.Tensor
    slot_id: int
    pool: "PinnedStagingPool"
    released: bool = False

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.pool._release(self.slot_id)

    def __enter__(self) -> "StagingLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class PinnedStagingPool:
    """Bounded CPU staging slots with a safe pageable fallback.

    A full Engram table must not be pinned.  The pool checks ``RLIMIT_MEMLOCK``
    when possible and falls back to pageable CPU tensors if PyTorch/ROCm cannot
    allocate pinned memory.  Callers can inspect ``pinned`` in benchmark logs.
    """

    def __init__(
        self,
        *,
        slots: int = 2,
        max_rows: int = 64,
        row_bytes: int,
        pin_memory: bool = True,
    ) -> None:
        if slots <= 0 or max_rows <= 0 or row_bytes <= 0:
            raise ValueError("slots, max_rows and row_bytes must be positive")
        self.slots = slots
        self.max_rows = max_rows
        self.row_bytes = row_bytes
        self.requested_pin_memory = pin_memory
        self.pinned = False
        bytes_requested = slots * max_rows * row_bytes
        if pin_memory:
            try:
                soft_limit, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
                if soft_limit != resource.RLIM_INFINITY and bytes_requested > soft_limit:
                    logger.warning(
                        "Engram staging request %d bytes exceeds RLIMIT_MEMLOCK %d; "
                        "using pageable staging",
                        bytes_requested,
                        soft_limit,
                    )
                    pin_memory = False
            except (AttributeError, OSError):
                pass

        self._slots: list[torch.Tensor] = []
        try:
            self._slots = [
                torch.empty(
                    (max_rows, row_bytes),
                    dtype=torch.uint8,
                    pin_memory=pin_memory,
                    device="cpu",
                )
                for _ in range(slots)
            ]
            self.pinned = bool(pin_memory)
        except (RuntimeError, AssertionError, NotImplementedError, OSError) as exc:
            if pin_memory:
                logger.warning("Pinned Engram staging unavailable (%s); using pageable memory", exc)
                self._slots = [
                    torch.empty((max_rows, row_bytes), dtype=torch.uint8, device="cpu")
                    for _ in range(slots)
                ]
            else:
                raise
        self._free = list(range(slots))
        self._condition = threading.Condition()

    def acquire(self, num_rows: int, *, timeout: float | None = None) -> StagingLease:
        if num_rows < 0 or num_rows > self.max_rows:
            raise ValueError(f"num_rows={num_rows} exceeds staging capacity {self.max_rows}")
        with self._condition:
            if timeout is None:
                while not self._free:
                    self._condition.wait()
            else:
                deadline = time.monotonic() + timeout
                while not self._free:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(remaining)
                if not self._free:
                    raise TimeoutError("timed out waiting for Engram staging slot")
            slot_id = self._free.pop()
        return StagingLease(self._slots[slot_id], slot_id, self)

    def _release(self, slot_id: int) -> None:
        with self._condition:
            if slot_id not in self._free:
                self._free.append(slot_id)
                self._condition.notify()

    def write_rows(self, lease: StagingLease, rows: Sequence[bytes]) -> int:
        if lease.pool is not self or lease.released:
            raise RuntimeError("lease is not active for this pool")
        if len(rows) > self.max_rows:
            raise ValueError("too many rows for staging slot")
        if any(len(row) != self.row_bytes for row in rows):
            raise ValueError("row byte width does not match staging pool")
        if rows:
            packed = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(
                len(rows), self.row_bytes
            )
            # ``from_numpy`` requires a writable view; copying here also keeps
            # the staging tensor independent from the temporary joined bytes.
            source = torch.from_numpy(packed.copy())
            lease.tensor[: len(rows)].copy_(source)
        return len(rows)

    def copy_to_device(
        self,
        lease: StagingLease,
        num_rows: int,
        device: torch.device | str,
        *,
        stream: Any = None,
    ) -> torch.Tensor:
        if lease.pool is not self or lease.released:
            raise RuntimeError("lease is not active for this pool")
        if num_rows < 0 or num_rows > self.max_rows:
            raise ValueError("invalid row count")
        target = torch.empty(
            (num_rows, self.row_bytes), dtype=torch.uint8, device=device
        )
        if stream is not None:
            with torch.cuda.stream(stream):
                target.copy_(lease.tensor[:num_rows], non_blocking=self.pinned)
        else:
            target.copy_(lease.tensor[:num_rows], non_blocking=self.pinned)
        return target


@dataclass
class EngramPrefetchResult:
    layer_id: int
    tensor_name: str
    requested_row_ids: tuple[int, ...]
    unique_row_ids: tuple[int, ...]
    requested_to_unique: tuple[int, ...]
    lease: StagingLease
    num_rows: int

    @property
    def host_tensor(self) -> torch.Tensor:
        return self.lease.tensor[: self.num_rows]

    def to_device(
        self, device: torch.device | str, *, stream: Any = None
    ) -> torch.Tensor:
        return self.lease.pool.copy_to_device(
            self.lease, self.num_rows, device, stream=stream
        )

    def release(self) -> None:
        self.lease.release()


class EngramPrefetcher:
    """Single-worker asynchronous host row prefetcher.

    The worker only performs mmap reads and CPU staging.  Device copies happen
    when the consumer calls ``EngramPrefetchResult.to_device`` so graph capture
    and stream ownership remain explicit to the future V4.1 runner.
    """

    def __init__(
        self,
        table: EngramHostTable,
        staging: PinnedStagingPool,
        *,
        max_workers: int = 1,
    ) -> None:
        self.table = table
        self.staging = staging
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="dsv41-engram"
        )
        self._closed = False
        self._close_lock = threading.Lock()

    @staticmethod
    def _deduplicate(row_ids: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        unique: list[int] = []
        positions: dict[int, int] = {}
        remap: list[int] = []
        for value in row_ids:
            value = int(value)
            position = positions.get(value)
            if position is None:
                position = len(unique)
                positions[value] = position
                unique.append(value)
            remap.append(position)
        return tuple(unique), tuple(remap)

    def submit(
        self,
        layer_id: int,
        row_ids: Sequence[int],
        *,
        tensor_name: str = "embed.weight",
    ) -> Future[EngramPrefetchResult]:
        with self._close_lock:
            if self._closed:
                raise RuntimeError("EngramPrefetcher is closed")
        requested = tuple(int(row_id) for row_id in row_ids)
        unique, remap = self._deduplicate(requested)

        def work() -> EngramPrefetchResult:
            lease = self.staging.acquire(len(unique))
            try:
                rows = self.table.read_rows_bytes(
                    layer_id, unique, tensor_names=(tensor_name,)
                )[tensor_name]
                self.staging.write_rows(lease, rows)
                return EngramPrefetchResult(
                    layer_id=layer_id,
                    tensor_name=tensor_name,
                    requested_row_ids=requested,
                    unique_row_ids=unique,
                    requested_to_unique=remap,
                    lease=lease,
                    num_rows=len(unique),
                )
            except BaseException:
                lease.release()
                raise

        return self._executor.submit(work)

    def fetch_sync(
        self,
        layer_id: int,
        row_ids: Sequence[int],
        *,
        tensor_name: str = "embed.weight",
    ) -> EngramPrefetchResult:
        return self.submit(layer_id, row_ids, tensor_name=tensor_name).result()

    def close(self, *, wait: bool = True) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def __enter__(self) -> "EngramPrefetcher":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "EngramHostTable",
    "EngramLayerHostTable",
    "EngramPrefetchResult",
    "EngramPrefetcher",
    "EngramRowMapping",
    "PinnedStagingPool",
    "RawMMapRowStore",
    "SafetensorsRowStore",
    "StagingLease",
]
