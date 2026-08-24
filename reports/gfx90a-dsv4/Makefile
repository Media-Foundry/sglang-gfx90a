REPORT := main

.PHONY: all check clean distclean

all: $(REPORT).pdf

$(REPORT).pdf: $(REPORT).tex
	latexmk -xelatex -interaction=nonstopmode -halt-on-error $(REPORT).tex

check:
	./check-tex-deps.sh

clean:
	latexmk -c $(REPORT).tex

distclean:
	latexmk -C $(REPORT).tex
