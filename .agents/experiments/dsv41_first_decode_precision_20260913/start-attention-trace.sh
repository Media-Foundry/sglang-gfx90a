#!/usr/bin/env bash
set -euo pipefail
cd /home/pc/Code/sglang
export MAX_TOTAL_TOKENS=32768 CONTEXT_LENGTH=32768
export FORWARD_HOOKS_JSON='[{"name":"attention-precision","target_modules":["model.embed_tokens","model.layers.4.*","model.layers.6.*"],"hook_factory":"scripts.rocm.dsv41_trace_hooks:make_hook","config":{"folder":"/tmp/dsv41-attention-precision-20260913","label":"attn","max_calls":6,"max_tensor_bytes":33554432,"capture_parameters":false,"attention_contract_layers":[4,6],"ranks":[0]}}]'
exec bash scripts/rocm/start_dsv41_correctness.sh
