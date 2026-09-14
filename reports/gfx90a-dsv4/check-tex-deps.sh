#!/usr/bin/env bash
set -euo pipefail

missing=0

for command_name in pdflatex latexmk kpsewhich; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "missing command: ${command_name}"
    missing=1
  fi
done

if command -v kpsewhich >/dev/null 2>&1; then
  for tex_file in \
    article.cls fontenc.sty helvet.sty courier.sty geometry.sty \
    amsmath.sty amssymb.sty booktabs.sty longtable.sty tabularx.sty \
    array.sty multirow.sty graphicx.sty xcolor.sty enumitem.sty \
    fancyhdr.sty listings.sty upquote.sty caption.sty tikz.sty \
    hyperref.sty cleveref.sty phvr8t.tfm pcrr8t.tfm; do
    if [[ -z "$(kpsewhich "${tex_file}")" ]]; then
      echo "missing TeX file: ${tex_file}"
      missing=1
    fi
  done
fi

if [[ "${1:-}" == "--figures" ]]; then
  for python_module in matplotlib seaborn pandas; do
    if ! "${PYTHON:-python}" -c "import ${python_module}" >/dev/null 2>&1; then
      echo "missing Python module: ${python_module}"
      missing=1
    fi
  done
fi

if (( missing )); then
  echo "See README.md for dependencies; no packages were installed."
  exit 1
fi

echo "pdfLaTeX commands, packages, and TeX-distributed fonts are available."
