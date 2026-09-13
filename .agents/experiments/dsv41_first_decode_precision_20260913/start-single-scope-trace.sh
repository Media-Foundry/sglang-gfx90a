#!/usr/bin/env bash
set -euo pipefail
cd /home/pc/Code/sglang
export MAX_TOTAL_TOKENS=32768 CONTEXT_LENGTH=32768
export SGLANG_DSV41_WO_A_INVARIANT=1
export FORWARD_HOOKS_JSON='[{"name":"single-scope-precision","target_modules":["model.embed_tokens","model.layers.1.self_attn.*","model.layers.1.self_attn","model.layers.1.input_layernorm"],"hook_factory":"scripts.rocm.dsv41_trace_hooks:make_hook","config":{"folder":"/tmp/dsv41-single-scope-precision-20260913","label":"single","max_calls":5,"max_tensor_bytes":33554432,"capture_parameters":true,"attention_contract_layers":[1]}}]'
exec bash scripts/rocm/start_dsv41_correctness.sh
