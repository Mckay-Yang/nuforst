# NUFROST Paper

- Main manuscript is `nufrost.tex` at the repo root. Treat `example/` as upstream IEEE template reference material, not part of the paper workflow.
- Build with `latexmk -pdf -interaction=nonstopmode -halt-on-error nufrost.tex` from the repo root. `latexmk` is installed and recognizes the paper as the only build target.
- There is no BibTeX or `.bib` flow here. References are maintained inline in `nufrost.tex` inside `thebibliography`.
- `IEEEtran.cls` is vendored in the repo root and is loaded locally by the manuscript. Do not switch to a system class file unless that is the explicit task.
- Preserve the current document class and journal wiring in `nufrost.tex`: it uses `\documentclass[letterpaper,journal]{IEEEtran}` plus explicit `\markboth`, `\IEEEpubid`, and `\IEEEpubidadjcol`.
- Figures are referenced by relative path from the root manuscript; the current external asset is `figures/accuracy.png`.
- Build artifacts are emitted into the repo root as `nufrost.{aux,fdb_latexmk,fls,log,pdf,synctex.gz}`. Do not hand-edit generated files.
- For verification after structural edits, rebuild and inspect `nufrost.log` for warnings. The current log is not clean: it already contains an `Underfull \hbox` warning, so treat that as pre-existing unless your change alters it.
