#!/usr/bin/env bash
# Measured original-V4 TP8 C16x8K checkpoint. Not a universal numerical guarantee.
set -euo pipefail
export SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER=1
export SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER_CHECK=0
export SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA=0
export SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA_CHECK=0
unset SGLANG_DSV4_DEBUG_PREMIX_OWNER_AUDIT_DIR
exec bash /home/pc/Code/sglang/.agents/experiments/dsv4_mhc_post_tiles_20260916/validated-launcher.sh
