# DeepSeek-V4-Flash on MI250: technical report v0.3

English report, updated September 14, 2026. The principal new result is the
single-instance TP8 native-AR matrix (C1/2/4/8/16/32/64), with separate prefill,
resident-decode, and whole-wave HTTP metrics. Historical TP4, strict DSpark,
approximate-target DSpark, and the new opt-in C32 consumer pilot are explicitly
separated. No new GPU measurements were performed to produce this report.

## Build the paper

Use **pdfLaTeX**, not XeLaTeX. Body text uses TeX-distributed Type 1 Helvetica;
the pre-generated Seaborn figures embed Arial. This avoids the earlier arXiv
`fontspec`/system-font failure. References use `thebibliography` inside `main.tex`:
there is no BibTeX/Biber, shell escape, or external style file.

```bash
cd reports/gfx90a-dsv4
make check
make
python check_report.py --compiled
make arxiv
```

`make` consumes the committed PDF figures and TeX inputs; it needs no Python or
GPU. The validation and packaging targets use only the Python standard library.
Output: `main.pdf`. The arXiv target creates `dsv4f_arxiv.tar.gz`, then extracts it
to a temporary directory and compiles it independently with pdfLaTeX until
references stabilize (up to four passes). It
rejects unresolved citations/references, missing glyphs, and overfull boxes. This
is local verification, not a claim that arXiv's remote build was tested.

For arXiv, upload **the source archive**, not `main.pdf`. Select **pdfLaTeX**;
the pre-generated figure PDFs are included. The archive excludes the compiled
article, auxiliary files, raw benchmark responses, plotting dependencies, and
Python scripts. The small frozen numerical snapshot is included for provenance.

Typical Ubuntu TeX dependencies (install only if `make check` reports missing):

```bash
sudo apt-get install --no-install-recommends \
  texlive-latex-extra texlive-pictures texlive-fonts-recommended latexmk
```

No environment creation or package installation is part of the build. Use the
existing conda environment; `PYTHON=/path/to/python make check` overrides Python.

## Regenerate data and figures

`data/results.json` freezes the three formal rounds per cell, the independent
ABBA samples, metadata hashes, and archive identities. `generated/` holds the
derived formal results table. The historical strict-DSpark ranges are rounded
ledger summaries, not reconstructed raw replicates or confidence intervals.

```bash
# Only when refreshing against the repository evidence:
make refresh-data
# Requires matplotlib, seaborn, pandas and Arial (or an explicit fallback):
make figures
make check
make
```

Normal/arXiv builds never access the parent repository. `prepare_report_data.py`
does so only when explicitly invoked; it checks source hashes and the numerical
summaries before freezing them. Plotting uses Arial when installed, otherwise
the named Liberation Sans fallback. Only left/bottom chart spines remain.
Actual replicate points and observed ranges are shown for current results; none
is represented as a confidence interval. Historical milestone charts are not
controlled cumulative ablations.

## Result scope and gaps

- Formal archive: `3d73a75923`; original weights, TP8/EP1/no-A2A, one allocated
  1,048,576-token logical KV pool, native graph tiers 1 through 64.
- P uses approximately 8K public-source input tokens and one output token;
  admission is capped at 16 requests. D uses approximately 512 input tokens,
  natural EOS, and at most 2048 outputs. Three rounds per cell, warmups excluded.
- Resident D is a common stream window, not a GPU-only timer or full HTTP rate.
  C1 uses its separately measured GEMV-on supplement; larger-tier HTTP cells
  retain the original GEMV-off drain. No all-on HTTP matrix is implied.
- C32 pilot: `fb2e39fca9`, 1044.61 to 1060.70 resident tok/s (+1.54%), default-off.
  This smaller ABBA screen does not replace the formal C32 cell (1044.32).
- Historical strict DSpark: 1127.48 resident tok/s with FP32 MHC fusion; not
  revalidated after the latest merges. Historical approximate-target 1.5k is
  neither strict verification nor native AR.
- A current matched TP4 matrix and a filled-1M-context test remain missing.
  Historical temporary files were not used to invent missing cells. The report
  does not claim universal numerical equivalence or a hardware-counter roofline.
- Benchmark services were stopped. API examples are not live-service claims.

Primary evidence lives under `.agents/experiments/dsv4_tp8_revalidation_20260913/`
and `.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/`. Source/manifest and
compressed archive SHA256 values are preserved in `data/results.json`.
