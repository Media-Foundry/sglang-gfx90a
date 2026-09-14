# Original V4 English report v0.3 — September 14, 2026

User requested the former TeX report be fully updated with the complete result.
This was a **report-only** task: no new inference measurement, model load, GPU
experiment, service launch, or package installation was performed.

## Evidence incorporated

`reports/gfx90a-dsv4/main.tex` now includes the formal single-instance TP8/EP1
native-AR matrix archived at `3d73a75923`, plus the independently scoped
default-off C32 down-consumer pilot `fb2e39fca9`.

| C | P input tok/s | Resident AR output tok/s | Whole HTTP output tok/s |
|--:|--:|--:|--:|
| 1 | 4676.39 | 87.60 | 86.39 |
| 2 | 4989.25 | 109.94 | 102.41 |
| 4 | 5265.43 | 188.71 | 148.50 |
| 8 | 5099.23 | 334.18 | 254.45 |
| 16 | 5171.36 | 608.23 | 422.51 |
| 32 | 5254.98 | 1044.32 | 680.61 |
| 64 | 5250.05 | 1334.24 | 848.14 |

P/resident D are three-round medians; HTTP aggregates measured waves. P uses
8K public-source inputs/one output/admission16; D uses 512-token inputs,
natural EOS/max2048 outputs and duration-driven resident windows. C1 is the
separate GEMV-on supplement; larger-tier HTTP retains GEMV-off drain. The C8
slow measured P round is preserved. A 1M logical pool was allocated, not filled
and certified at a million-token context.

The native C32 pilot's 1044.61→1060.70 (+1.54%) is separate from the formal
matrix. Its C1 89.72→90.02 is an unchanged-kernel isolation check, not a C1
improvement or replacement matrix result. Whole-model bitwise parity is not
claimed. Separate 8K C32 empty-tile ABBA (3.599x resident) and C1 GEMV ABBA
(+10.32%) retain their distinct inputs/metrics and cannot be multiplied.

Historical TP4 ~74.5 is native short-probe HTTP, not DSpark/TP8. Historical
strict TP8 DSpark 1127.48 (FP32 fusion) is not revalidated after the latest merge.
Old TP4 ~1.5k anchor-only verification is explicitly approximate, not strict.
Historical TP8 6.42k P used ~2304-token inputs and a 131072 pool, not current
8K/1M. Missing retained matched TP4 matrix cells are not invented.

MI250 (not MI250X), eight GCDs/four accelerators, host DIMM fault reported fixed,
current NUMA interleave, sampled vs exact memory peaks, external dirty AIter/CK
dependencies, and stopped services are all distinguished in the revision.

## Build and validation actually run

- `make refresh-data`: checked SHA256/size of both compressed evidence archives
  and **all 278 formal + 92 pilot members**; froze three-round medians, ABBA
  values, and source hashes into `data/results.json` and the generated table.
- `make figures`: regenerated eight PDF/PNG plots using installed **Arial**,
  only left/bottom spines. Current plots show actual rounds or waves; historical
  DSpark ranges are rounded ledger min/max, not fabricated replicates or CIs.
  Historical truncated ABBA bars were replaced with explicitly expanded dots.
- `make check`, Python syntax checks for all four report scripts, shell syntax
  check, `make`, and `check_report.py --compiled` passed.
- `pdffonts main.pdf`: every listed font embedded; body Type 1 Helvetica-compatible
  fonts, figure Arial TrueType; no Type 3 font or host fontspec dependency.
- Rendered all 18 pages with pdftoppm; visually reviewed pages 1, 5, 8–12, and
  18 plus the four new figures. Fixed annotation overlap and small-panel text.
- `package_arxiv.py`: whitelist includes **14 files** (4 TeX, 8 PDF figures,
  frozen JSON, README). Extracted the packaged bytes into an independent temporary
  directory, compiled with `pdflatex -no-shell-escape` until stable: **3 passes**.
  No undefined citations/references, missing glyphs, or overfull boxes. Local
  TeX Live **2023/Debian**; arXiv's remote TeX Live 2025 was not tested.
- `git diff --check` passed. No model/runtime code was edited. The existing dirty
  `cuda_graph_runner_memory_usage.pickle` and unrelated experiment files were not
  included or reverted.

## Deliverables

- `reports/gfx90a-dsv4/main.pdf`: 18 pages, 452345 bytes at this build;
  SHA256 `8225f6b75927903be5fab1ce03471f7d6f01a53953db7583c4571e70fba6a829`.
- `reports/gfx90a-dsv4/dsv4f_arxiv.tar.gz`: 295686 bytes;
  SHA256 `852cd8570585b402dff6cba099309cc70d72ecca3f1a74acda50a0d85c23e2b7`.
- Source archive excludes the compiled article, auxiliaries and Python; figures
  are pre-generated. **Select pdfLaTeX** for arXiv. Normal `make` needs neither
  Python nor a GPU; explicit data/figure refresh targets are separate.

Earlier source archive remains recoverable from Git history. This update is
on `sync/media-foundry-gfx90a`; the remote branch was at `fb2e39fca9` before the
report commit. Do not overwrite the independently advanced remote `main`.
