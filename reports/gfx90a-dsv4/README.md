# DeepSeek-V4-Flash gfx90a technical report

The report uses XeLaTeX and an Arial-compatible system sans-serif font. It intentionally uses
`listings` and an inline bibliography, so it does not require shell escape,
Pygments, BibTeX, or Biber.

## Required Ubuntu packages

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends \
  texlive-xetex texlive-latex-extra texlive-pictures \
  texlive-fonts-recommended latexmk fonts-liberation fonts-noto-core
```

Check dependencies and build:

```bash
cd reports/gfx90a-dsv4
make check
make
```

The comparison charts are generated with Seaborn. Regenerate them directly with:

```bash
python generate_figures.py
```

The script uses Arial when it is installed and otherwise uses the metric-compatible
Liberation Sans fallback. Every chart removes the top and right spines.

For the genuine Microsoft Arial font on Ubuntu, optionally install the core-font
package and rebuild the figures and report:

```bash
sudo apt-get install ttf-mscorefonts-installer
fc-cache -f
python generate_figures.py
make
```

The output is `main.pdf`. Generated LaTeX files are ignored by the report-local
`.gitignore`.
