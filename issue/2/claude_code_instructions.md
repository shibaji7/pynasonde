# Claude Code Instructions for CADI Issue #2

You are continuing CADI support for pynasonde issue #2:

https://github.com/shibaji7/pynasonde/issues/2

## Current Development Rule

Do not commit unless the user explicitly asks. This CADI work is part of the pynasonde 2.0 development track and should not trigger a PyPI release or version bump yet.

## Read First

Read these files before making changes:

- `issue/2/cadi_development_summary.md`
- `issue/2/cadi_format_notes.md`
- `issue/2/cadi_next_steps.md`
- `tmp/CADI/mdx2txt_cliV2.py`
- `tmp/CADI/mdx2csv_cliV2.py`
- `pynasonde/digisonde/cadi/reader.py`
- `pynasonde/digisonde/cadi/extractor.py`
- `pynasonde/digisonde/digi_plots.py`
- `tests/test_cadi_reader.py`

If working on O/X mode separation, also read:

- `pynasonde/vipir/analysis/polarization.py`

## Preserve Current APIs

Do not break these APIs:

- `CadiReader(...).parse()`
- `CadiExtractor(...).to_dataframe_raw()`
- `CadiExtractor(...).to_dataframe_products()`
- `CadiExtractor.extract_CADI(...)`
- `CadiExtractor.load_CADI_files(...)`
- `CadiIonogram.add_power_ionogram(...)`
- `CadiIonogram.add_doppler_ionogram(...)`

## Scientific Caution

Current `doppler_bin` is a bin/index product, not calibrated Hz.

Do not label it as physical Doppler frequency unless a calibration path is provided. Any `doppler_hz` implementation must accept explicit calibration parameters or return `NaN` when calibration is unknown.

Current CADI products do not provide direct O/X labels. O/X separation must be implemented as a classifier/inference layer and should default to `unknown` unless defensible criteria are available.

## Testing

Use targeted tests while iterating:

```bash
NUMBA_DISABLE_JIT=1 pytest -q tests/test_cadi_reader.py
NUMBA_DISABLE_JIT=1 pytest -q tests/test_digisonde_examples.py::test_run_cadi_basic_example
NUMBA_DISABLE_JIT=1 pytest -q tests/test_digisonde_examples.py::test_run_cadi_products_example
NUMBA_DISABLE_JIT=1 pytest -q tests/test_digisonde_examples.py::test_run_cadi_interferometry_plot_example
```

The `NUMBA_DISABLE_JIT=1` prefix is for this local workstation environment. The underlying issue is a broader import-time `timezonefinder`/`numba` cache problem, not a CADI parser problem.

## Git Hygiene

The working tree may contain unrelated user changes. Stage only files related to CADI issue #2 if the user later asks for a commit.

Do not revert unrelated files.

Likely CADI-related paths:

- `pynasonde/digisonde/cadi/`
- `pynasonde/digisonde/digi_plots.py`
- `pynasonde/digisonde/__init__.py`
- `pynasonde/__init__.py`
- `examples/digisonde/cadi_*.py`
- `tests/test_cadi_reader.py`
- `tests/test_digisonde_examples.py`
- `issue/2/`
