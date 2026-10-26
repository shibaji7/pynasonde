# Pynasonde 2.0 setup plan

## Context
pynasonde is at 1.3.0 (tags v1.0.0–v1.3.0; HEAD is 24 commits past v1.3.0). We want a 2.0 line where
stable releases have matching stable docs on Read the Docs, users can install only the modules they need,
tests/examples are verified, and docs are refreshed. 2.0 changes are selective and incremental, with **no
removed code and no broken 1.x imports**. Each phase below is its own commit/PR, done one at a time.

Baseline (verified): 904 tests pass, 19 skip (missing RIQ data), 0 fail. Docs = MkDocs Material + mkdocstrings on RTD.

## Key findings driving the plan
- `pynasonde/__init__.py` eagerly imports `digisonde`, `vipir`, `webhook` -> any install needs all deps.
- `digisonde/parsers/sao.py`, `edp.py`, `digi_utils.py` import `TimeZoneConversion` from `vipir/ngi/utils.py` (drags toml/pytz/timezonefinder).
- Version in 3 places (`setup.py` 1.3.0 + `use_scm_version`, `pynasonde/__init__.py`, pyproject dynamic). `setup.py` and `pyproject.toml` disagree on extras; deps are hard-pinned (numpy==1.26.4 etc.).
- Undeclared deps: fdb, joblib, scienceplots (module-level import in `digi_utils.py`), matlab.engine. `training` extra has no code behind it; nrlmsise00 declared but unused.
- docs/requirements.txt is stale (numpy 1.23, sphinx, flask...). No changelog, no version switcher, README has no install section.
- Deprecation noise: `importlib.resources.path` at `vipir/ngi/utils.py:197`. Aliases `RSF_IONOGRAM_SETTINGS`/`SBF_IONOGRAM_SETTINGS` must stay.
- Docstrings: mixed numpy/google; gaps in `digisonde/cadi` (9), `vipir/analysis` (11), `matlab_lib`, `webhook.py`; type hints weak in `vipir/ngi`, `digisonde/datatypes`.

## Scope decisions (agreed with user)
- **No commits without explicit agreement.** Phase 1 and the analysis-scope work below are in the working tree, uncommitted, awaiting review.
- ~~vipir/analysis in 2.0: only `polarization` and `scaler`, plus their examples and docs~~ (done in working tree)
  - ~~Other analysis modules (absorption, inversion, spread_f, irregularities, nextyz, es_imaging): files untouched, 1.x imports still resolve via lazy `__getattr__`, hidden from 2.0 `__all__`, nav and API index~~
  - ~~`vipir-analysis` extra trimmed to `[]` (polarization/scaler need only core deps); sklearn/skimage/opencv moved into `vipir` extra~~
  - ~~New example docs: `docs/examples/vipir/polarization.md`, `scaler.md`; analysis index rewritten for 2.0 scope~~
  - Open: Es-imaging example page and the digisonde modules still need a 2.0 scope decision (Es example removed from nav, file kept).

## Phase 1 — Packaging + lazy imports (working tree, uncommitted)
1. ~~Move metadata/deps into `pyproject.toml`; `setup.py` reduced to a stub; package-data declared; MANIFEST.in covers `.m`~~ (wheel build verified)
2. ~~Single-source version via setuptools-scm: `version_file = pynasonde/_version.py` (gitignored), `__init__` imports it with `0.0.0+unknown` fallback; Makefile `check`/`release` step 1 updated to use the git tag~~ (verified: wheel, installed and source-tree versions all track the tag). Open: `CITATION.cff` / `codemeta.json` versions still manual.
3. ~~Core deps with lower bounds; extras `digisonde`, `digisonde-db`, `ocr`, `vipir`, `vipir-analysis`, `time`, `web`, `plots`, `matlab`, `all`, `docs`, `dev`~~ (`training` kept)
   - ~~Verified in clean venvs: core (analysis polarization/scaler run), `digisonde` (389 passed), `vipir` (358 passed)~~
   - ~~Verified in clean venvs: `ocr`, `web`, `digisonde-db`, `dev` (full suite 936 passed). Fixed `dev`/`all` missing `vipir`.~~
   - ~~Found + fixed: Matplotlib 3.11 placed `clabel` labels far outside the date axis (render overflow in `SaoSummaryPlots` isodensity). Out-of-range labels are now dropped; the temporary `<3.11` cap is lifted. Full suite passes on 3.11.2.~~
4. ~~Lazy `pynasonde/__init__.py`, `vipir/__init__.py`, `vipir/analysis/__init__.py`, `digisonde/__init__.py`, `digisonde/parsers/__init__.py` (PEP 562; OCR parser no longer drags OpenCV into other parsers)~~ Still eager (deferred, low value): `vipir/ngi/__init__.py`, `vipir/riq/__init__.py`.
5. ~~Move `TimeZoneConversion` to `pynasonde/timeutils.py`; re-export from `vipir/ngi/utils.py`~~ ~~`scienceplots` guard: `digi_utils` was already guarded; fixed the real hard dependency in `vipir/ngi/utils.setsize`~~
6. ~~Replace deprecated `importlib.resources.path`~~
7. ~~`tests/test_lazy_imports.py` added~~ ~~`needs_scipy` guards in `tests/test_vipir_analysis.py`~~ ~~webhook tests skip without `web` extra~~
   Full suite: 936 passed, 19 skipped, 0 failed.

## Phase 2 — Docs versioning + changelog
1. `.readthedocs.yaml`: install via `pip install .[docs]`; add `docs` extra (mkdocs, material, mkdocstrings-python, glightbox, awesome-pages); delete stale `docs/requirements.txt` (or make it `-e .[docs]`).
2. Make older tags buildable: note that v1.x tags carry their own config; in RTD UI activate v1.0.0, v1.1.0, v1.2.x, v1.3.0 (+ later v2.0.x), set `stable` = latest tag, `latest` = main. Check each old tag builds; if one fails, add a docs-only fixup via RTD per-version override rather than rewriting tags.
   **Version dropdown:** Read the Docs adds its own version/language flyout (bottom-left, or in the Material header via RTD addons) listing every *active* version, so no extra plugin is needed. Each version is built from its own tag, so v1.x docs keep their 1.x install/usage text and v2.0 docs show the extras-based install. Early check in this phase: inspect `.readthedocs.yaml`/`mkdocs.yml` at each old tag (only HEAD was inspected so far) and confirm each builds; a tag that lacks a config needs an RTD-side override, never a moved tag.
   **Install/usage text per version:** the 2.0 install page must say "pip install pynasonde" (core), "pip install 'pynasonde[vipir]'" etc., and link back to the 1.3 docs with a note that 1.x installed everything; add an "Upgrading from 1.x" box (all 1.x imports still work; for former full install use `pynasonde[all]`).
3. Add `CHANGELOG.md` (reconstruct 1.0–1.3 from git log/tags), a "What's new in 2.0 / migration" docs page, and install page documenting extras. Expand README with install + extras + docs link. Update `Makefile release` to include docs/version check.
4. Sync version in `CITATION.cff` and `codemeta.json`.

## Phase 3 — Test and example hardening
1. CI: matrix Python 3.11/3.12; add a job that installs each extra in a clean venv and runs an import smoke test; a job on minimal core install running only core tests.
2. Mark tests needing optional deps with `pytest.importorskip`; keep the 19 data-based skips but document them.
3. Examples: add `tests/test_examples_smoke.py` that runs bundled-data examples (DFT/SKY) headless; add data guards (clear message) to examples with hardcoded `/media`/`/home` paths; label the 2026-01-19 and IP-shock scripts as event-specific.
4. Address warnings (numpy/fork/polyfit) where they're ours.

## Phase 4 — Docstring/type pass
1. Standardize on Google style (matches mkdocstrings config); convert numpy-style files opportunistically, no logic changes.
2. Fill gaps: `digisonde/cadi`, `vipir/analysis`, `matlab_lib`, `webhook.py`; add types in `vipir/ngi`, `digisonde/datatypes`; add `py.typed` and lightweight mypy config (non-blocking).
3. Verify with `mkdocs build --strict` and add a docs-build CI job.

## Selecting what is "new in 2.0"
After phases 1–4, pick a short list of functional 2.0 items one at a time (user choice each time); each follows: add new API alongside old, emit `DeprecationWarning` on old, keep tests for both, document in CHANGELOG.

## Verification
- `pytest -q --no-cov` -> still 904+ passed, 0 failed.
- In fresh venvs: `pip install .` then `python -c "import pynasonde"`; `pip install .[digisonde]`, `.[vipir]`, `.[all]` each import their module; core install must not import fdb/cv2/sklearn.
- `python -W error::DeprecationWarning -c "import pynasonde.digisonde, pynasonde.vipir"` on `[all]`.
- `mkdocs build --strict`; check RTD builds for each activated tag and that `stable`/`latest` resolve.
- `python -m build` and `twine check dist/*`; package data present in wheel.
