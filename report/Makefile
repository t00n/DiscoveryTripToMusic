
report.pdf: report.tex

%.pdf: %.tex
	pdflatex $<
	# bibtex $(subst .tex,,$<)
	# pdflatex $<
	# pdflatex $<

clean:
	rm -f *.aux *.log *.out

mrproper: clean
	rm -f *.pdf
