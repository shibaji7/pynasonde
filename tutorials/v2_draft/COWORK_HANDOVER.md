# Handover: draft the pynasonde 2.0 SoftwareX paper

Paste the **Prompt** into Claude Cowork and attach the files listed under *Attach*. The
**Facts pack** below it was verified against the repository, PyPI, GitHub and Read the Docs on
2026-10-26; Cowork should treat it as the source of truth and flag anything it cannot confirm.

## Attach

- `1-s2.0-S235271102600110X-main.pdf` : the published v1 paper (SoftwareX 34 (2026) 102617)
- `softwarex_draft.tex` : the initial v2 draft (same folder)
- `figures/pynasonde_v2_overview.pdf` (+ `.png`) : the new one-page overview figure
- `CHANGELOG.md`, `ROADMAP.md`, `CONTRIBUTING.md` from the repository root

## Prompt

> You are helping me write the SoftwareX paper for **pynasonde v2.0** (Python library for
> ionosonde data: VIPIR, Digisonde, CADI, plus the DIDBase and DriftBase databases). I have attached
> the published v1 paper, my initial v2 draft (`softwarex_draft.tex`), a one-page overview figure,
> and the changelog/roadmap/contributing files. A verified **Facts pack** follows this prompt.
>
> **Do these, in order:**
> 1. Read the v1 paper and my draft. Keep the SoftwareX structure (Code metadata, Motivation and
>    significance, Software description, Illustrative examples, Impact, Conclusions).
> 2. **Search GitHub issues** on `shibaji7/pynasonde` (open and closed, plus discussions and pull
>    requests) and summarise what users ask for and where they struggle. Three issues exist today
>    (see the Facts pack); check for new ones.
> 3. **Find papers that cite pynasonde** (the v1 paper, DOI 10.1016/j.softx.2026.102617, and any
>    that mention `pynasonde` or the repository): search Google Scholar, NASA ADS, Crossref,
>    OpenAlex, Semantic Scholar, arXiv, theses, and AGU/EGU/URSI abstracts. For each, say **how
>    they use it** (instrument, module, purpose). Give links. Never invent a citation; list what
>    you could not verify separately.
> 4. Use the Facts pack for the "what is new in v2" content, and put the overview figure in the
>    architecture subsection. Say plainly which pieces are old (v1.x) and which are new in 2.0.
> 5. Add a **short section on maintenance since v1.0**: after v1.0.0 the code had small fixes,
>    several 1.x releases were made **because users asked** (see the issue table), and we added
>    versioned documentation per release (Read the Docs), a changelog, contributing guide,
>    roadmap, CI per install extra, and a docs-versus-code check.
> 6. In **Impact**, describe real use: a planetary-physics course tried the SAO example and
>    could not find the example SAO files (we treat this as a gap: ship small sample files);
>    Cornell University uses it for space-science study. Ask me before naming people; otherwise
>    say "an instructor" / "a university group".
>
> **Rules:** every claim must trace to the Facts pack or to a source you cite. Do not call
> unfinished or out-of-scope features stable. Do not say 1.x installed its dependencies (it did
> not). Mark anything uncertain as `TODO(author)`. Return the updated `.tex`, a table of
> citing works, an issue summary, and a list of open questions for me.

## Facts pack (verified 2026-10-26)

### Releases (git tags; GitHub release dates differ for 1.1.0/1.2.0)

| Version | Date | Note |
|---------|------|------|
| v1.0.0 | 2025-10-29 | SoftwareX submission; the version described by the v1 paper |
| v1.1.0 / v1.2.0 | tags 2026-03-19 / 03-21 (GitHub releases 2026-03-23) | MATLAB bridge (1.1); echo extractor and ionogram filter (1.2.0) |
| v1.2.1 | 2026-04-06 | |
| v1.2.2 | 2026-05-29 | **Multi-record SAO day-files**, made for issue #3 the same day |
| v1.3.0 | 2026-06-26 | **CADI MD2/MD4**, GRM parser, DIDBase connector; CADI requested in issue #2 |
| 2.0.0 | not tagged yet | everything below is on `main` and pushed |

### What is new in 2.0 (compared with 1.3.0)

- **Digisonde:** a new autoscaler (`pynasonde.digisonde.scaler`): canonical echo schema for RSF/SBF/MMM,
  RFI filters, Viterbi tracing, hmF2, presets, solar-regime helpers; `GrmSplitter` frame slicing.
- **Databases:** a **DriftBase** connector (drift/SKY files, velocities), next to the DIDBase connector (1.3).
  Credentials come from a git-ignored `.env`; opt-in live login tests (`make test-live`) pass for both.
- **VIPIR:** O/X polarization classifier and ionogram scaler are documented, tested and run on the
  core install (they first appeared in 1.2.x). Other analysis modules (absorption, inversion, spread-F,
  irregularities, NeXtYZ, Es imaging) still exist but are **not** in the 2.0 documented scope.
- **Packaging:** lean core install plus per-module extras (`digisonde`, `digisonde-scaler`, `digisonde-db`,
  `ocr`, `vipir`, `web`, `all`, ...), lazy imports (every 1.x import path still works), version from the git tag.
- **Documentation and process:** versioned docs per release on Read the Docs, CHANGELOG, CONTRIBUTING,
  ROADMAP, an install page that says which extra each class needs, `make check-docs` (docs checked
  against the code), CI with one job per install extra and Python 3.11 and 3.12.
- **Numbers:** about 980 tests; Codecov line coverage 53.94%; 54 API directives and 190 pynasonde imports in
  docs and examples all resolve.

### Packaging defect found (important for honesty in the paper)

The published 1.0.0 to 1.3.0 packages on PyPI **declare no runtime dependencies**. A clean
`pip install pynasonde==1.3.0` installs only pynasonde and `import pynasonde` fails. It went unnoticed
because the maintainer's conda environment already had the packages. A user reported it (issue #1);
2.0 fixes it.

### GitHub issues and users (3 issues today; 6 stars, 2 forks, 3 watchers)

| # | Opened | By | Topic | Outcome |
|---|--------|----|-------|---------|
| 1 | 2026-02-26 | @mhogan-nwra | PyPI install fails with missing `loguru` etc.; asked for `dependencies` per PEP 631 | A fix attempt followed in March, but PyPI metadata shows the defect persisted through 1.3.0; fixed in 2.0. **Still open**: close after the 2.0 release |
| 2 | 2026-05-21 | @AbdollahMasoud | Add CADI MD4 support | Delivered in v1.3.0 (MD2 and MD4). Calibration caveat: XL/YL, angle-of-arrival and Doppler-Hz products are not final. **Still open** |
| 3 | 2026-05-29 | @RR-ion | Cannot read a day-style `.SAO` file | Delivered in v1.2.2 the same day; reporter confirmed it works. Closed |

Direct user contacts (not on GitHub, ask before naming):

- An instructor of a *Physics of the Planets* course asked for the SAO files behind the docs example
  "SAO: Scaled Ionogram Height Profiles and F2-Layer Diagnostics". The students could not find them on
  UMass Lowell, NOAA or other servers. **The SAO example's data is not shipped in the repository**
  (only small DFT, SKY and RSF samples are); this is an acknowledged gap.
- Cornell University is using pynasonde for space-science study (told by the maintainer; details to be
  confirmed).

### The v1 paper and the repository

- v1: S. Chakraborty, T. Bullett, A. Barjatya, J. Mabie, SoftwareX 34 (2026) 102617, DOI
  10.1016/j.softx.2026.102617 (open access, CC BY-NC).
- Repository: https://github.com/shibaji7/pynasonde (MIT) · Docs: https://pynasonde.readthedocs.io/ ·
  PyPI: https://pypi.org/project/pynasonde/ (latest published: 1.3.0).

### Caveats to state accurately

- The 1.x documentation is frozen per tag and has small errors (for example an example imports a class
  that does not exist). From 1.2.0 on, every documented API page resolves; v1.0.0 and v1.1.0 document a
  few modules that were not yet in those releases.
- The 161 MB WI937 RIQ sample used by the VIPIR examples is not stored in the repository.
- v1.3.0 contained an NN-inversion prototype (25 files) that was removed from `main` on 2026-06-28; it
  is still in the v1.3.0 tag. Decide whether to mention it only as future work.
- Typing is partial (mypy baseline of several hundred issues), so no `py.typed` marker yet.
