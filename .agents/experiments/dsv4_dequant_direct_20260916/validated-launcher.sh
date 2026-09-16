#!/usr/bin/env bash
# Measured TP8 C16x8K original-V4 native-AR checkpoint, 1M logical KV.
set -euo pipefail
export SGLANG_DSV4_DEBUG_CK_DIRECT_DEQUANT=1
export SGLANG_DSV4_DEBUG_CK_DIRECT_DEQUANT_CHECK=0
exec bash /home/pc/Code/sglang/.agents/experiments/dsv4_premix_owner_20260916/validated-launcher.sh
