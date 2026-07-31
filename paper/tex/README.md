# Manuscript build (CAS / AIM)

Single-column Elsevier CAS manuscript for *Artificial Intelligence in Medicine*.

## Files

| File | Role |
|------|------|
| `paper-last.tex` | Canonical manuscript (inline TikZ figures) |
| `cas-refs.bib` | Bibliography |
| `cas-sc.cls`, `cas-common.sty` | Elsevier CAS single-column class |
| `elsarticle-num-names.bst` | Numbered Vancouver-style names (required by AIM) |
| `cas-model2-names.bst` | Author–year style (not used; shipped with CAS) |
| `thumbnails/` | CAS email/social icons required by `\maketitle` |
| `fig_pipeline.tex` | **Duplicate** of the overview figure — do not edit alone; see header comment |
| `cover_letter.md` | Submission cover letter |
| `highlights.txt` | Plain-text highlights for Editorial Manager |
| `paper-last.pdf` | Compiled PDF (gitignored under `paper/tex/*.pdf`; copied into `../submission/`) |

## Compile

Requires a TeX engine that can resolve CAS + `pgfplots`/`tikz` (tested with [Tectonic](https://tectonic-typesetting.github.io/)):

```bash
cd paper/tex
tectonic -p --keep-logs paper-last.tex
# or: pdflatex paper-last && bibtex paper-last && pdflatex paper-last && pdflatex paper-last
```

Submission-ready copies live in [`../submission/`](../submission/).
