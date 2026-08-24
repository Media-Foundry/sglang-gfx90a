# DeepSeek-V4-Flash gfx90a technical report

The report uses XeLaTeX and system Noto CJK fonts. It intentionally uses
`listings` and an inline bibliography, so it does not require shell escape,
Pygments, BibTeX, or Biber.

## Required Ubuntu packages

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends \
  texlive-xetex texlive-lang-chinese texlive-latex-extra \
  texlive-pictures latexmk fonts-noto-cjk
```

Check dependencies and build:

```bash
cd reports/gfx90a-dsv4
make check
make
```

The output is `main.pdf`. Generated LaTeX files are ignored by the report-local
`.gitignore`.
