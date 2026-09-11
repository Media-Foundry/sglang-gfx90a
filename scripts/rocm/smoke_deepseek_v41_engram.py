#!/usr/bin/env python3
"""Inspect real V4.1 Engram headers and a few host-RAM rows.

This command is intentionally payload-light: it reads only selected rows from
Engram tensors and never creates a full GPU table.  It should be run after the
ModelScope download has completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experimental.deepseek_v41.engram_host import (  # noqa: E402
    EngramHostTable,
    EngramPrefetcher,
    PinnedStagingPool,
)
from experimental.deepseek_v41.metadata import audit_checkpoint  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--tp-size", default=8, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--rows", default=3, type=int)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)

    report = audit_checkpoint(args.model_dir, tp_size=args.tp_size)
    if not report.get("engram_payload_complete", False) and not args.allow_incomplete:
        print(
            "Engram payload is not complete; rerun after shards 47/48 finish "
            "or pass --allow-incomplete.",
            file=sys.stderr,
        )
        return 2

    table = EngramHostTable.from_safetensors_index(
        args.model_dir,
        tp_size=args.tp_size,
        rank=args.rank,
        allow_incomplete=args.allow_incomplete,
    )
    print(json.dumps(table.describe(), sort_keys=True))
    if not table.layers:
        table.close()
        return 2

    try:
        for layer_id, layer in sorted(table.layers.items()):
            print(
                f"layer={layer_id} rows={layer.mapping.global_rows} "
                f"range=[{layer.mapping.global_start},{layer.mapping.global_end})"
            )
            for name, store in layer.stores.items():
                shape = getattr(store, "shape", None)
                print(
                    f"  {name}: dtype={store.dtype} rows={store.rows} "
                    f"row_bytes={store.row_bytes} shape={shape}"
                )
            # q/k/wkv are static operator tensors, not table rows.  Probe only
            # the small q/k/scale payloads here; leave the 157 MiB wkv weight
            # lazy and never materialize it merely for a smoke test.
            static_probe = tuple(
                name
                for name in ("q_weight", "k_weight", "wkv.scale")
                if name in layer.static_tensor_names
            )
            if static_probe:
                static = layer.read_static_bytes(static_probe)
                print(
                    "  static_probe="
                    + json.dumps(
                        {
                            name: {
                                "bytes": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest(),
                            }
                            for name, payload in static.items()
                        },
                        sort_keys=True,
                    )
                )
            if "embed.weight" not in layer.stores:
                continue
            last = layer.mapping.global_rows - 1
            candidates = [0, last // 2, last]
            ids = tuple(dict.fromkeys(candidates[: max(args.rows, 1)]))
            rows = layer.read_rows_bytes(ids, tensor_names=("embed.weight",))["embed.weight"]
            digest = hashlib.sha256(b"".join(rows)).hexdigest()
            print(f"  embed_probe_ids={list(ids)} sha256={digest}")

            row_bytes = layer.stores["embed.weight"].row_bytes
            staging = PinnedStagingPool(
                slots=1,
                max_rows=max(len(ids), 1),
                row_bytes=row_bytes,
                pin_memory=True,
            )
            with EngramPrefetcher(table, staging) as prefetcher:
                result = prefetcher.fetch_sync(layer_id, ids)
                print(
                    f"  staged_rows={result.num_rows} pinned={staging.pinned} "
                    f"shape={tuple(result.host_tensor.shape)}"
                )
                result.release()
    finally:
        table.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
