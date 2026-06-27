## pynasonde v1.3.0

### New Features

#### CADI Support (resolves [#2](https://github.com/shibaji7/pynasonde/issues/2))
- Added `pynasonde.digisonde.cadi` subpackage for Canadian Advanced Digital
  Ionosonde (CADI) MD2/MD4 binary files.
- `CadiReader` — low-level binary decoder for headers, detections, and I/Q samples.
- `CadiExtractor` — user-facing layer with `to_dataframe_raw()`,
  `to_dataframe_products()`, and `load_CADI_files()` for parallel batch loading.
- `CadiIonogram` — plotting layer in `digi_plots.py` with `add_power_ionogram()`
  and `add_doppler_ionogram()` methods.
- Products include amplitude, phase, baseline phase differences (`dphi_12`,
  `dphi_13`, `dphi_23`), and coherence values across all receiver pairs.

#### GRM Parser
- New `GrmSplitter` parser for GRM-format files (`pynasonde.digisonde.parsers.grm`).

#### RSF/SBF Base Class
- Introduced `RsfSbfBinaryBlockExtractor` in `parsers/_rsf_sbf_base.py`,
  eliminating ~400 lines of duplicated code between RSF and SBF parsers.
- `RSF_SBF_IONOGRAM_SETTINGS` is now a single source of truth in `digi_utils.py`;
  backwards-compatible aliases kept in `rsf.py` and `sbf.py`.

#### MMM Parser Rewrite
- `ModMaxExtractor` completely rewritten — previous implementation only parsed
  block 0, referenced non-existent attributes, and was effectively non-functional.
- Now correctly iterates all blocks and returns a usable DataFrame with columns
  `frequency_mhz`, `range_km`, `amplitude_dB`, `polarization`, `doppler_channel`.

#### EDP Parser
- `load_EDP_files()` was a stub that silently returned `None`. Now fully
  implemented using the centralized `load_files_to_dataframe()` utility.

#### DIDBase Connector
- New `DidBaseConnector` in `pynasonde.digisonde.didbase`.

### Improvements

- **`digi_utils.py`**: Added `apply_filename_metadata()`, `load_files_to_dataframe()`,
  `collect_files()`, `RSF_SBF_IONOGRAM_SETTINGS`, and shared byte-reading utilities,
  eliminating ~200 lines of duplicated `__init__` logic across all parsers.
- **Docstrings**: Full Google-style docstring coverage across all 21 digisonde
  modules. `mkdocs.yml` updated with `docstring_style: google` and nav entries
  for SBF, MMM, GRM parsers, MMM datatypes, and DIDBase Connector.
- **New example docs**: `docs/examples/digisonde/mmm.md` and
  `docs/examples/digisonde/sao_multi.md`.

### Bug Fixes

- `sbf.py`: Fixed `NameError` — `two_bytes` used before assignment in `extract()`.
- `mmm.py`: Fixed `break` after block 0 — only the first block was ever parsed.
- `edp.py`: Fixed `load_EDP_files()` silently returning `None`.
- `digi_utils.py`: Fixed `load_dtd_file()` context-manager bug with deprecated
  `importlib.resources.path()`.
- `digi_utils.py`: Fixed `load_station_csv()` using deprecated
  `importlib.resources.path()`.
- `digi_utils.py`: Fixed `SettingWithCopyWarning` in `get_gridded_parameters()`.
- `dvl.py`: Fixed mutable default argument `folders=["tmp/..."]` with hardcoded
  developer path.
- `raw/raw_plots.py`: Replaced bare `print()` with `logger.debug()`.
- `digi_utils.py`: Made `scienceplots` import optional in `setsize()`.

### Packaging

- `netCDF4` added to `install_requires` (hard import in `raw_parse.py` was missing).
- `SciencePlots` moved from `install_requires` to `extras_require["plots"]`
  (`pip install pynasonde[plots]`).
- `scikit-learn` added to `extras_require["dev"]`.

### CI / Workflow

- Bumped `actions/checkout` → `v5` (Node 24).
- Bumped `actions/setup-python` → `v6` (Node 24).
- Bumped `codecov/codecov-action` → `v4`.
- Added `tesseract-ocr` apt install step.
- Added `[plots]` extra to CI install command.
