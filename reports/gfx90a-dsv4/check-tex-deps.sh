#!/usr/bin/env bash
set -euo pipefail

missing=0

for command_name in pdflatex bibtex latexmk kpsewhich; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "missing command: ${command_name}"
    missing=1
  fi
done

if command -v kpsewhich >/dev/null 2>&1; then
  for tex_file in \
    acmart.cls ACM-Reference-Format.bst libertine.sty zi4.sty newtxmath.sty \
    binhex.tex fontenc.sty amsmath.sty booktabs.sty tabularx.sty \
    graphicx.sty xcolor.sty listings.sty upquote.sty subcaption.sty \
    tikz.sty hyperref.sty cleveref.sty xurl.sty; do
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

echo "PPoPP/acmart, BibTeX, and template-default font dependencies are available."
