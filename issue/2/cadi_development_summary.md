# CADI Development Summary for Issue #2

Issue: https://github.com/shibaji7/pynasonde/issues/2

Scope: CADI support for the pynasonde 2.0 development track. This work is currently intended for GitHub development only, not a PyPI release.

## Implemented Files

- `pynasonde/digisonde/cadi/reader.py`
- `pynasonde/digisonde/cadi/extractor.py`
- `pynasonde/digisonde/cadi/__init__.py`
- `pynasonde/digisonde/digi_plots.py`
- `pynasonde/digisonde/__init__.py`
- `pynasonde/__init__.py`
- `examples/digisonde/cadi_basic.py`
- `examples/digisonde/cadi_products.py`
- `examples/digisonde/cadi_interferometry_plot.py`
- `tests/test_cadi_reader.py`
- `tests/test_digisonde_examples.py`

## Current CADI API

`CadiReader` is the low-level MD2/MD4 binary reader. It decodes:

- `CadiHeader`
- `CadiDetection`
- `CadiDataset`
- frequency table in Hz
- per-detection I/Q samples for all receivers

`CadiExtractor` is the user-facing extraction layer. It exposes:

- `extract()`
- `to_dataframe_raw()`
- `to_dataframe_products()`
- `extract_CADI(...)`
- `load_CADI_files(...)`

`CadiIonogram` is the plotting layer in `digi_plots.py`. It exposes:

- `add_power_ionogram(...)`
- `add_doppler_ionogram(...)`

## DataFrame Products

Raw extraction returns one row per detected CADI echo with:

- site and source file metadata
- time index and per-record timestamp
- frequency index, `frequency_hz`, `frequency_mhz`
- height flag and `height_km`
- `doppler_flag`
- receiver I/Q columns: `rxN_i_raw`, `rxN_q_raw`, `rxN_i`, `rxN_q`

Product extraction adds:

- `doppler_bin`
- `rxN_amp`
- `rxN_phase_rad`
- `rxN_phase_deg`
- `mean_power_db`
- baseline phase differences: `dphi_12_rad`, `dphi_12_deg`, etc.
- grouped coherence values: `coh_12`, `coh_13`, `coh_23`

## Validation

Targeted CADI validation has passed with:

```bash
NUMBA_DISABLE_JIT=1 pytest -q tests/test_cadi_reader.py \
  tests/test_digisonde_examples.py::test_run_cadi_basic_example \
  tests/test_digisonde_examples.py::test_run_cadi_products_example \
  tests/test_digisonde_examples.py::test_run_cadi_interferometry_plot_example
```

Latest result observed during development:

```text
10 passed, 1 warning
```

The `NUMBA_DISABLE_JIT=1` environment variable is used locally because the broader package import can trigger a `timezonefinder`/`numba` cache issue in this workstation environment.

## Important Constraints

- Do not commit until the CADI feature set is reviewed and finalized.
- Do not bump package version for this CADI track yet.
- Do not create a PyPI release for this work yet.
- Keep this tied to issue #2 and the pynasonde 2.0 development track.
