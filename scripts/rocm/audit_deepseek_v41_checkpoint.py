#!/usr/bin/env python3
"""Audit a DeepSeek-V4.1 checkpoint without loading tensor payloads.

This command is safe while ModelScope is still downloading shards.  It reads
config/index JSON and safetensors headers only; use ``--require-complete`` to
turn an incomplete download into a non-zero exit status.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keep the script runnable from a source checkout without installing SGLang.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experimental.deepseek_v41.metadata import audit_checkpoint  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--index", dest="index_path", type=Path, default=None)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument(
        "--no-header-validation",
        action="store_true",
        help="only inspect JSON manifests; do not parse present shard headers",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="return status 2 unless all indexed shard payloads are present and valid",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="optional path for the JSON audit report",
    )
    parser.add_argument(
        "--meta-smoke",
        action="store_true",
        help="also instantiate the payload-free meta-device model and validate [2, 8] input",
    )
    parser.add_argument(
        "--require-engram",
        action="store_true",
        help="return status 2 until all indexed Engram shards/headers are available",
    )
    parser.add_argument(
        "--engram-host-smoke",
        action="store_true",
        help="build the lazy RAM-backed Engram table and print its mapping",
    )
    return parser


def _print_summary(report: dict) -> None:
    print(f"status: {report.get('status')}")
    print(f"model_dir: {report.get('model_dir')}")
    counts = report.get("counts", {})
    for name in (
        "indexed_shards",
        "complete_tensors",
        "missing_tensors",
        "missing_shards",
        "incomplete_shards",
        "engram_tensors",
        "engram_missing_tensors",
        "mtp_tensors",
        "vision_tensors",
        "backbone_experts_from_index",
        "mtp_experts_from_index",
    ):
        if name in counts:
            print(f"{name}: {counts[name]}")
    config = report.get("config", {})
    if config:
        print(
            "config: "
            f"arch={config.get('architectures')} "
            f"model_type={config.get('model_type')} "
            f"layers={config.get('num_hidden_layers')} "
            f"nextn={config.get('num_nextn_predict_layers')} "
            f"experts={config.get('n_routed_experts')}"
        )
    layout = report.get("attention_layout_observed", {})
    if layout:
        modes = {}
        for info in layout.values():
            mode = info.get("observed_mode")
            modes[mode] = modes.get(mode, 0) + 1
        print(f"attention_layout_observed: {modes}")
    if "engram_payload_complete" in report:
        print(f"engram_payload_complete: {report['engram_payload_complete']}")
    if report.get("engram_shards"):
        for layer, info in report["engram_shards"].items():
            print(
                f"engram layer {layer}: global_rows={info['global_rows']} "
                f"rows_per_rank(padded)={info['rows_per_tp_rank_with_padding']}"
            )
    for warning in report.get("warnings", ()):
        print(f"warning: {warning}", file=sys.stderr)
    for error in report.get("errors", ()):
        print(f"error: {error}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_checkpoint(
        args.model_dir,
        tp_size=args.tp_size,
        index_path=args.index_path,
        validate_headers=not args.no_header_validation,
    )
    _print_summary(report)
    if args.meta_smoke and report.get("status") != "invalid":
        try:
            import torch
            from experimental.deepseek_v41.meta_model import DeepSeekV41MetaModel
            model = DeepSeekV41MetaModel.from_json(
                args.model_dir / "config.json", tp_size=args.tp_size
            )
            output = model(torch.zeros((2, 8), dtype=torch.long, device="meta"))
            print(f"meta_smoke: shape={tuple(output.shape)} device={output.device}")
            report["meta_smoke"] = {
                "shape": list(output.shape),
                "device": str(output.device),
                "dtype": str(output.dtype),
            }
        except Exception as exc:
            print(f"meta_smoke_error: {exc}", file=sys.stderr)
            report["meta_smoke_error"] = str(exc)
            if args.require_complete:
                return 1
    if args.engram_host_smoke:
        try:
            from experimental.deepseek_v41.engram_host import EngramHostTable
            table = EngramHostTable.from_safetensors_index(
                args.model_dir, tp_size=args.tp_size, allow_incomplete=True
            )
            print(f"engram_host_smoke: {json.dumps(table.describe(), sort_keys=True)}")
            table.close()
        except Exception as exc:
            print(f"engram_host_smoke_error: {exc}", file=sys.stderr)
            report["engram_host_smoke_error"] = str(exc)
            if args.require_engram:
                return 1
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"json_report: {args.json_out}")
    if args.require_engram and not report.get("engram_payload_complete", False):
        return 2
    if args.require_complete and report.get("status") != "ready":
        return 2
    return 0 if report.get("status") != "invalid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
