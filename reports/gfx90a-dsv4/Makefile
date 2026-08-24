REPORT := main
FIGURES := figures/decode_progress.pdf figures/prefill_progress.pdf \
	figures/prefill_abba.pdf figures/moe_microbenchmark.pdf

.PHONY: all check clean distclean

all: $(REPORT).pdf

$(REPORT).pdf: $(REPORT).tex $(FIGURES)
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(REPORT).tex

$(FIGURES): generate_figures.py
	python generate_figures.py

check:
	./check-tex-deps.sh

clean:
	latexmk -c $(REPORT).tex

distclean:
	latexmk -C $(REPORT).tex
	rm -f figures/*.pdf figures/*.png
