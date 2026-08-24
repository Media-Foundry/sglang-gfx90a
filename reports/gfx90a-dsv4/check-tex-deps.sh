#!/usr/bin/env bash
set -euo pipefail

missing=0

for command_name in xelatex latexmk kpsewhich; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "missing command: ${command_name}"
    missing=1
  fi
done

if command -v kpsewhich >/dev/null 2>&1; then
  for tex_file in \
    ctexart.cls fontspec.sty geometry.sty amsmath.sty booktabs.sty \
    longtable.sty tabularx.sty multirow.sty listings.sty tikz.sty \
    pgfplots.sty hyperref.sty cleveref.sty; do
    if [[ -z "$(kpsewhich "${tex_file}")" ]]; then
      echo "missing TeX file: ${tex_file}"
      missing=1
    fi
  done
fi

for font_name in \
  "Noto Serif" "Noto Sans" "Noto Sans Mono" \
  "Noto Serif CJK SC" "Noto Sans CJK SC" "Noto Sans Mono CJK SC"; do
  if ! fc-match "${font_name}" | grep -qi "Noto"; then
    echo "missing font: ${font_name}"
    missing=1
  fi
done

if (( missing )); then
  cat <<'EOF'

Install the Ubuntu packages with:
  sudo apt-get update
  sudo apt-get install --no-install-recommends \
    texlive-xetex texlive-lang-chinese texlive-latex-extra \
    texlive-pictures latexmk fonts-noto-cjk
EOF
  exit 1
fi

echo "TeX commands, packages, and fonts are available."
