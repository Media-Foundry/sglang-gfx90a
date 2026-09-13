#!/usr/bin/env bash
set -euo pipefail
cd /home/pc/Code/sglang
export MAX_TOTAL_TOKENS=32768 CONTEXT_LENGTH=32768
export SGLANG_DSV41_WO_A_INVARIANT=1
export FORWARD_HOOKS_JSON='[{"name":"invariant-woa-precision","target_modules":["model.embed_tokens","model.layers.*.input_layernorm","model.layers.*.post_attention_layernorm","model.layers.*.self_attn","model.layers.*.mlp","model.layers.*.engram","model.norm"],"hook_factory":"scripts.rocm.dsv41_trace_hooks:make_hook","config":{"folder":"/tmp/dsv41-invariant-woa-precision-20260913","label":"invwoa","max_calls":5,"max_tensor_bytes":33554432,"capture_parameters":false,"ranks":[0]}}]'
exec bash scripts/rocm/start_dsv41_correctness.sh
