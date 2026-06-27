# CADI Next Steps

This file captures the remaining development work for CADI support after the current binary reader, extractor, product, and plotting work.

## 1. Doppler Calibration

Current state:

- `doppler_flag` is decoded from the file.
- `doppler_bin` is exposed as a product column.
- No physical Doppler Hz conversion is implemented yet.

Next implementation:

- Add configurable conversion from bin to signed bin.
- Add optional conversion from signed bin to Hz.
- Keep raw `doppler_bin` unchanged.

Possible API:

```python
CadiExtractor.extract_CADI(
    file,
    product="products",
    doppler_zero_bin=None,
    doppler_bin_spacing_hz=None,
    doppler_positive_direction=1,
)
```

Potential output columns:

- `doppler_signed_bin`
- `doppler_hz`

Do not invent calibration constants. If unknown, leave `doppler_hz` absent or `NaN`.

## 2. O/X Mode Separation

Current state:

- CADI products include power, phase, baseline phase difference, and coherence.
- No explicit O/X labels are available from the current MD4 decode.

Recommended path:

- Read `pynasonde/vipir/analysis/polarization.py`.
- Reuse the analysis pattern where appropriate, but do not copy VIPIR assumptions directly.
- Implement CADI-specific O/X work as an adapter/classifier layer.

Candidate output columns:

- `mode`: `O`, `X`, or `unknown`
- `mode_confidence`
- `mode_method`

Important: default should remain `unknown` until the classifier is scientifically defensible.

## 3. Interferometry / AoA

Current state:

- Baseline phase differences are available: `dphi_12`, `dphi_13`, `dphi_23`.
- Coherence values are available: `coh_12`, `coh_13`, `coh_23`.
- No antenna baseline geometry or AoA solver is implemented.

Next implementation:

- Add a CADI baseline geometry config object.
- Add phase-bias correction hooks.
- Add AoA solver only when baseline geometry and wavelength are available.

Suggested files:

- `pynasonde/digisonde/cadi/interferometry.py`
- `pynasonde/digisonde/cadi/config.py`

## 4. Plotting Expansion

Current state:

- `CadiIonogram.add_power_ionogram(...)`
- `CadiIonogram.add_doppler_ionogram(...)`

Next useful methods:

- `add_phase_ionogram(..., zparam="dphi_12_deg")`
- `add_coherence_ionogram(..., zparam="coh_12")`
- optional multipanel helper for power/Doppler/phase/coherence.

## 5. Cleanup Before Final Commit

Before committing CADI work:

- Run formatting if required by repo convention.
- Run targeted CADI tests.
- Run parser/import smoke tests.
- Review dirty tree and stage only CADI/issue-2 related files.
- Do not include unrelated generated figures, nn inversion changes, VIPIR changes, or local config churn.

Suggested targeted tests:

```bash
NUMBA_DISABLE_JIT=1 pytest -q tests/test_cadi_reader.py
NUMBA_DISABLE_JIT=1 pytest -q tests/test_digisonde_examples.py::test_run_cadi_basic_example
NUMBA_DISABLE_JIT=1 pytest -q tests/test_digisonde_examples.py::test_run_cadi_products_example
NUMBA_DISABLE_JIT=1 pytest -q tests/test_digisonde_examples.py::test_run_cadi_interferometry_plot_example
```
