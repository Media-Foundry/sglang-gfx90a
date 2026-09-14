REPORT := main
PYTHON ?= python
FIGURES := figures/decode_progress.pdf figures/prefill_progress.pdf \
	figures/prefill_abba.pdf figures/moe_microbenchmark.pdf \
	figures/tp8_pd_matrix.pdf figures/native_abba.pdf \
	figures/down_consumer_abba.pdf figures/strict_dspark_history.pdf
TEX_INPUTS := $(wildcard sections/*.tex generated/*.tex)

.PHONY: all check figures refresh-data arxiv clean

all: $(REPORT).pdf

$(REPORT).pdf: $(REPORT).tex $(TEX_INPUTS) $(FIGURES)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(REPORT).tex

# Ordinary builds consume committed figures; arXiv never runs Python.
figures:
	./check-tex-deps.sh --figures
	$(PYTHON) generate_figures.py

refresh-data:
	$(PYTHON) prepare_report_data.py

check:
	./check-tex-deps.sh
	$(PYTHON) check_report.py

arxiv: all check
	$(PYTHON) check_report.py --compiled
	$(PYTHON) package_arxiv.py

clean:
	latexmk -c $(REPORT).tex
