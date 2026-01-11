echo "\printbibliography" >> paper_v4.tex
pdflatex paper_v4.tex
biber paper_v4
pdflatex paper_v4.tex
pdflatex paper_v4.tex
