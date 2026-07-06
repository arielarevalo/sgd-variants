#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null

rm -f main.aux main.bbl main.blg main.log main.out main.toc main.synctex.gz

echo "report/main.pdf ready"
