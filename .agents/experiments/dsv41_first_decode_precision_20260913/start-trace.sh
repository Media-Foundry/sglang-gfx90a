#!/usr/bin/env bash
# Numerical diagnostic only: synchronous CPU snapshots invalidate timing.
set -euo pipefail
cd /home/pc/Code/sglang
export MAX_TOTAL_TOKENS=32768 CONTEXT_LENGTH=32768
export FORWARD_HOOKS_JSON='[{"name":"first-decode-precision","target_modules":["model.embed_tokens","model.layers.0.*","model.layers.1.*","model.layers.*.input_layernorm","model.layers.*.post_attention_layernorm","model.layers.*.self_attn","model.layers.*.mlp","model.layers.*.engram","model.norm","logits_processor"],"hook_factory":"scripts.rocm.dsv41_trace_hooks:make_hook","config":{"folder":"/tmp/dsv41-first-decode-precision-20260913","label":"firstdecode","max_calls":8,"max_tensor_bytes":33554432,"capture_parameters":false,"ranks":[0]}}]'
exec bash scripts/rocm/start_dsv41_correctness.sh
