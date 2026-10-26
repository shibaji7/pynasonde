# Changelog

All notable changes to pynasonde are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/). Documentation for each release is
published on [Read the Docs](https://pynasonde.readthedocs.io/) (use the version
selector, or `/en/v1.3.0/`, `/en/stable/`, `/en/latest/`).

## [2.0.0] - unreleased

2.0 is a selective update: every 1.x import path keeps working.

### Added
- Per-module install extras: `digisonde`, `digisonde-scaler`, `digisonde-db`,
  `ocr`, `vipir`, `vipir-analysis`, `time`, `web`, `plots`, `matlab`, `all`,
  `docs`, `dev`. See [Installing Pynasonde](docs/user/install.md).
- `pynasonde.digisonde.scaler`: DataFrame-native Digisonde autoscaler (schema
  normalisation, RFI filters, Viterbi tracing, hmF2 scaling, presets, solar
  regime helpers) with an RSF example and API reference pages.
- `GrmSplitter.load_dataframes(frame_slice=...)` to load a subset of ionograms.
- `pynasonde.timeutils.TimeZoneConversion` (shared by Digisonde and VIPIR).
- Docs: examples for `PolarizationClassifier` and `IonogramScaler`; the VIPIR ionogram
  filter example also saves the decoded RIQ power figure, and its page now matches
  the current output.
- Version selector in the docs header (lists the versions active on Read the Docs and
  opens the same page in the chosen version; `docs/javascripts/version-selector.js`).
- `CONTRIBUTING.md` and `ROADMAP.md`, linked from the README.
- Credentials via a git-ignored `.env` (template: `.env.example`) and an opt-in live
  DIDBase login test (`make test-live`).
- Developer tooling: colourful `make` targets (`env`, `install-dev`, `test`,
  `format`, `docs`, ...) running in the `pynasonde` conda environment.

### Changed
- **Lean core install.** `pip install pynasonde` now installs only numpy,
  pandas, matplotlib, loguru, toml and tqdm. Use `pip install "pynasonde[all]"`
  for the full set that 1.x installed. Dependency pins are now lower bounds.
- Public names resolve lazily (PEP 562): `import pynasonde` no longer imports
  every subpackage or its optional dependencies. Existing imports such as
  `from pynasonde import SaoExtractor` are unchanged.
- The 2.0 `vipir.analysis` scope is `polarization` and `scaler`. The other
  analysis modules are untouched and still importable, but are not covered by
  2.0 docs, examples or extras (see the 1.3 docs for them).
- Version is single-sourced from the git tag via setuptools-scm.
- `pyproject.toml` is the single metadata source (`setup.py` is a stub).
- `examples/digisonde/rsf_direction_ionogram.py` is portable: station and data directory
  come from `RSF_EXAMPLE_STN` / `RSF_EXAMPLE_DIR` (default: bundled WS833 files).
- The `digisonde` extra now includes `joblib` (used by the Digisonde examples).
- Docs build uses `pip install .[docs]`; the stale `docs/requirements.txt` is gone.
- One-time isort/black formatting of the code base.

### Fixed
- `SaoSummaryPlots` isodensity plots failed to save with Matplotlib 3.11
  (out-of-range contour labels).
- `vipir.ngi.utils.setsize` required `scienceplots`; it is now optional.
- Deprecated `importlib.resources.path` usage in `load_toml`.
- Importing a Digisonde parser no longer requires OpenCV (only the OCR image
  parser does).

## [1.3.0] - 2026-06-26
- CADI MD2/MD4 support (`pynasonde.digisonde.cadi`: `CadiReader`, `CadiExtractor`,
  `CadiIonogram`).
- New `GrmSplitter` parser; `DidBaseConnector`; shared `RsfSbfBinaryBlockExtractor`.
- `ModMaxExtractor` (MMM) rewritten; `load_EDP_files()` implemented.
- Google-style docstring coverage across the Digisonde modules; several bug fixes
  (SBF, MMM, EDP, DVL, `digi_utils`).
- Full notes: [`issue/release_notes_v1.3.0.md`](issue/release_notes_v1.3.0.md).

## [1.2.2] - 2026-05-29
## [1.2.1] - 2026-04-06
## [1.2.0] - 2026-03-21
## [1.1.0] - 2026-03-19
Release notes for these versions were not written at the time; see the
[commit history](https://github.com/shibaji7/pynasonde/compare/v1.0.0...v1.3.0).

## [1.0.0] - 2025-10-29
Initial release for the SoftwareX submission.

[2.0.0]: https://github.com/shibaji7/pynasonde/compare/v1.3.0...main
[1.3.0]: https://github.com/shibaji7/pynasonde/compare/v1.2.2...v1.3.0
[1.2.2]: https://github.com/shibaji7/pynasonde/compare/v1.2.1...v1.2.2
[1.2.1]: https://github.com/shibaji7/pynasonde/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/shibaji7/pynasonde/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/shibaji7/pynasonde/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/shibaji7/pynasonde/releases/tag/v1.0.0
