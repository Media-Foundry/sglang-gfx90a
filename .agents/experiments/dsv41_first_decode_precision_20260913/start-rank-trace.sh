#!/usr/bin/env bash
set -euo pipefail
cd /home/pc/Code/sglang
export MAX_TOTAL_TOKENS=32768 CONTEXT_LENGTH=32768
export FORWARD_HOOKS_JSON='[{"name":"rank-attention-precision","target_modules":["model.embed_tokens","model.layers.4.self_attn.*","model.layers.6.self_attn.*","model.layers.4.self_attn","model.layers.6.self_attn"],"hook_factory":"scripts.rocm.dsv41_trace_hooks:make_hook","config":{"folder":"/tmp/dsv41-rank-attention-precision-20260913","label":"rankattn","max_calls":5,"max_tensor_bytes":33554432,"capture_parameters":true,"attention_contract_layers":[4,6]}}]'
exec bash scripts/rocm/start_dsv41_correctness.sh
