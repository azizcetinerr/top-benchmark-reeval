# AIM submission package

Ready-to-upload bundle for *Artificial Intelligence in Medicine* (Elsevier CAS, single-column).

## Upload checklist

1. **Manuscript PDF:** `paper-last.pdf` (29 pages, CAS `cas-sc`)
2. **LaTeX sources:** `paper-last.tex`, `cas-refs.bib`, `cas-sc.cls`, `cas-common.sty`, `elsarticle-num-names.bst`, `thumbnails/`
3. **Highlights:** paste from `highlights.txt` (5 items, ≤85 characters each)
4. **Cover letter:** `cover_letter.md` (convert to PDF/DOCX if Editorial Manager requires)
5. **Optional supplementary:** `SUPPLEMENTARY_NOTE.md` (points to the public GitHub repo)

## Reproduce numbers

From the repository root: `bash regenerate_all.sh`  
Verified against stored predictions: `paper/verify_paper_tables.py` (tables match `paper-last.tex`).

## Build locally

```bash
cd paper/tex
tectonic -p paper-last.tex
```
