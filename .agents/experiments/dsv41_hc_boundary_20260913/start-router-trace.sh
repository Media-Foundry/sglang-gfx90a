#!/usr/bin/env bash
set -euo pipefail
cd /home/pc/Code/sglang
export MAX_TOTAL_TOKENS=32768 CONTEXT_LENGTH=32768
export SGLANG_DSV41_WO_A_INVARIANT=1 SGLANG_DSV41_ATTN_INVARIANT=1
export SGLANG_DSV41_ROUTER_INVARIANT=1
export SGLANG_DSV41_HC_STATS_INVARIANT=1 SGLANG_DSV41_ENGRAM_GATE_INVARIANT=1
export FORWARD_HOOKS_JSON='[{"name":"router","target_modules":["model.embed_tokens","model.layers.*.input_layernorm","model.layers.*.post_attention_layernorm","model.layers.*.self_attn","model.layers.*.mlp","model.layers.*.engram","model.norm"],"hook_factory":"scripts.rocm.dsv41_trace_hooks:make_hook","config":{"folder":"/tmp/dsv41-router-precision-20260913","label":"router","max_calls":10,"capture_calls":[0,6,8],"max_tensor_bytes":33554432,"capture_parameters":false,"ranks":[0],"attention_contract_layers":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]}}]'
exec bash scripts/rocm/start_dsv41_correctness.sh
