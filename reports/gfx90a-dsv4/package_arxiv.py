#!/usr/bin/env python3
"""Package literal TeX dependencies and verify a clean pdfLaTeX build."""
import gzip
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile

from check_report import ROOT, check, check_log, tex_inputs


def main():
    check(compiled=True)
    tex, figures = tex_inputs(ROOT)
    paths = sorted(tex | figures | {Path('data/results.json'), Path('references.bib'), Path('main.bbl')})
    note = ('Compile main.tex with pdfLaTeX until references stabilize (2-4 passes).\n'
            'All figure PDFs are included.\n'
            'Uses acmart [sigplan,10pt,review] and ACM-Reference-Format.\n'
            'main.bbl is included; BibTeX is needed only to regenerate references.\n'
            'No Python, system fonts, shell escape, or Biber required.\n'
            'The compiled article is deliberately excluded from this source archive.\n'
            'data/results.json records evidence provenance and numerical summaries.\n')
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w') as archive:
            files = [(str(path), (ROOT / path).read_bytes()) for path in paths]
            files.append(('00README.txt', note.encode()))
            for name, raw in files:
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                info.mode = 0o644
                info.mtime = 0
                archive.addfile(info, io.BytesIO(raw))
    payload = buf.getvalue()
    # Verify exactly the packaged bytes, not an existing build's auxiliaries.
    with tempfile.TemporaryDirectory(prefix='dsv4-arxiv-check-') as directory:
        root = Path(directory)
        with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
            archive.extractall(root, filter='data')
        previous_aux = None
        for number in range(1, 5):
            result = subprocess.run(
                ['pdflatex', '-no-shell-escape', '-interaction=nonstopmode',
                 '-halt-on-error', 'main.tex'], cwd=root,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.returncode:
                raise RuntimeError(f'Clean bundle pass {number} failed:\n{result.stdout}')
            print(f'Clean extracted source: pdfLaTeX pass {number} completed.')
            auxiliary = tuple((root / f'main.{ext}').read_bytes() for ext in ('aux', 'out'))
            if auxiliary == previous_aux:
                break
            previous_aux = auxiliary
        check_log(root)
    target = ROOT / 'dsv4f_arxiv.tar.gz'
    target.write_bytes(payload)
    print(f'PASS: {len(paths) + 1} source files; {len(payload)} bytes; '
          f'SHA256 {hashlib.sha256(payload).hexdigest()}\n{target}')


if __name__ == '__main__':
    main()
