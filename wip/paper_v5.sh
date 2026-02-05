emacs paper_v5.org --batch -f org-latex-export-to-latex --kill
sed -i 's/\\documentclass\[11pt\]{article}/\\documentclass{ieeeqccl}/' paper_v5.tex
sed -i 's/\\end{document}/\\printbibliography\n\\end{document}/' paper_v5.tex
pdflatex paper_v5.tex
biber paper_v5
pdflatex paper_v5.tex
pdflatex paper_v5.tex
