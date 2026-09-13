#!/usr/bin/env bash
set -euo pipefail
cd /home/pc/Code/sglang
export MAX_TOTAL_TOKENS=32768 CONTEXT_LENGTH=32768
export SGLANG_DSV41_WO_A_INVARIANT=1 SGLANG_DSV41_ATTN_INVARIANT=1
export FORWARD_HOOKS_JSON='[{"name":"hc-boundary","target_modules":["model.embed_tokens"],"hook_factory":"scripts.rocm.dsv41_boundary_trace:make_hook","config":{"folder":"/tmp/dsv41-hc-boundary-20260913","layers":[12,13,20,21,22],"ranks":[0],"max_calls":10}}]'
exec bash scripts/rocm/start_dsv41_correctness.sh
