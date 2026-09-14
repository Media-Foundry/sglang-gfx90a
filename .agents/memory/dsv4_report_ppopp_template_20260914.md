# PPoPP-template migration — Siming HUANG / HKUST(GZ)

User requested the exact supplied workshop template and new author metadata.
Source ZIP: `reports/Differentiable_Programming_Workshop_at_PPoPP_24_Template.zip`.
SHA256 `8ae9f4a69be3ad1aaffd151e6827c506de9f1122cb9ee08188dd1713b45d7da7`.
It contains a sample main.tex and bibfile.bib, not a private document class.

## Implemented

- Exact class/options: `\documentclass[sigplan,10pt,review]{acmart}`.
- Retained template topmatter: `printfolios=true,printccs=false,printacmref=false`,
  `setcopyright{none}`, and `ACM-Reference-Format` bibliography style.
- Author **Siming HUANG**; institution **HKUST(GZ)**, Guangzhou, China. No email
  or ORCID invented. Visible title/header and XMP `dc:creator` checked.
- No custom geometry, body font, page style, line spacing, or class patches.
  Removed article-class TOC and custom Helvetica/fancyhdr setup. acmart owns
  two columns, 10pt body, margins, headings, captions and red review line numbers.
- Default Libertine/newtxmath/Inconsolata fonts; existing figure Arial retained.
- Cross-column floats replace incompatible full-width single-column floats and
  longtables. Added figure descriptions and repaired long text/code wrapping.
  Small wording changes preserve the measurement claims and caveats.
- References moved to `references.bib`; BibTeX generates standard ACM references.
  SGLang/vLLM/MegaBlocks entries cite their verified arXiv versions rather than
  invent missing proceedings fields. Removed an uncited white-paper entry.
- Conference identification is **September 2026 Technical Report**, not a false
  claim that these later results appeared at DiffProg/PPoPP'24. Format compliance
  does not imply submission, acceptance, or workshop page-limit compliance.
- Updated Makefile, dependency check, README, static checker, and source packager.
  Bundle includes the `.bib` and generated `main.bbl`; no Python/GPU is needed
  to compile it. It uses standard installed acmart, not a bundled imitation class.

## Dependencies / checks actually run

The user installed `texlive-publishers` and `texlive-fonts-extra`, version
2023.20240207-1. Installed acmart is **2.03 (2024/02/04)**, from TeX Live 2023.
newtxmath additionally needed `binhex.tex`. Without sudo, downloaded matching
Ubuntu `texlive-plain-generic` using apt, extracted it in a task-specific /tmp
directory, and installed **only the unmodified binhex.tex** into
`/home/pc/texmf/tex/generic/kastrup/binhex.tex` (previously absent).
SHA256 `69bbed0fb67874d2c60a4c277b6c350e7b5774c6dbc6686b77fbb40f895f3395`.
The regular sudo alternative is documented: install texlive-plain-generic.
Other downloaded temporary archives were not installed over system files.

- `make check`, Python syntax checks, shell syntax check, `latexmk -pdf`,
  `check_report.py --compiled`, and `git diff --check` passed.
- No unresolved citations/references, missing glyphs, or overfull h/v boxes.
  BibTeX has no warnings. Two expected **acmart topmatter** warnings remain:
  final ACM reference and CCS requirements, because the supplied review policy
  disables that display. These are documented, not misreported as citation errors.
- `pdffonts` verified all fonts embedded; body uses template fonts and figures
  retain Arial; no Type 3 fonts.
- Rendered and visually inspected all **13 pages**. Rechecked final pages 11–13
  after keeping the two-line parser listing together and adding appendix context.
- `make arxiv` packaged **16 files**, then extracted the archive and compiled it
  independently in **two pdfLaTeX passes** using its included main.bbl.
  Local TeX Live 2023 only; arXiv remote TeX Live 2025 is not claimed tested.
- Frozen numerical JSON, generated matrix, and eight figure PDFs unchanged.
  No GPU measurement, service launch, model/runtime change, or invented result.

## Final artifacts at this build

- `reports/gfx90a-dsv4/main.pdf`: **13 pages**, 757404 bytes;
  SHA256 `afb0cdfc4367fa046b95199324d168ff9b4d9f371257bfded5eabf7da0599cb1`.
- `reports/gfx90a-dsv4/dsv4f_arxiv.tar.gz`: 297899 bytes;
  SHA256 `445d62c961a65cc532f610d6fe22597be2fa1b4aef35a0f413cbc101f45f3dad`.

The supplied ZIP and unrelated dirty files are preserved. Changes belong on
`sync/media-foundry-gfx90a`, not the independently advanced remote main branch.
The previous 18-page edition remains recoverable at `853e92f598`.
