# What's New in 2.0

<div class="hero">
  <h3>A leaner install, with every 1.x import still working</h3>
  <p>
    pynasonde 2.0 is a selective update. Install only the modules you need,
    and keep your existing scripts unchanged.
  </p>
</div>

!!! info "Looking for the 1.x docs?"
    Use the version menu to switch, or open the
    [1.3 documentation](https://pynasonde.readthedocs.io/en/v1.3.0/). The
    [changelog](https://github.com/shibaji7/pynasonde/blob/main/CHANGELOG.md)
    lists every release.

## Upgrading from 1.x

`pip install pynasonde` now installs a **lean core**. To get everything that
1.x installed, use the `all` extra:

```bash
pip install --upgrade "pynasonde[all]"
```

Or install just what you use, for example `pip install "pynasonde[digisonde]"`
or `pip install "pynasonde[vipir]"`. See [Choose What to Install](install.md#choose-what-to-install).

Your code does not need to change:

- All 1.x import paths still work, including `from pynasonde import SaoExtractor`
  and `from pynasonde.vipir.analysis import AbsorptionAnalyzer`.
- `TimeZoneConversion` moved to `pynasonde.timeutils`; the old
  `pynasonde.vipir.ngi.utils.TimeZoneConversion` still works.
- If you see `ModuleNotFoundError: No module named 'sklearn'` (or `cv2`,
  `scipy`, …), install the extra for the module you are using.

## What changed

| Area | 2.0 |
|------|-----|
| Install | Lean core plus per-module extras; dependency pins are now minimum versions |
| Imports | Lazy: `import pynasonde` no longer imports every subpackage |
| Versioning | Single-sourced from the git tag (`pynasonde.__version__`) |
| Digisonde | New autoscaler (`pynasonde.digisonde.scaler`); `GrmSplitter.load_dataframes(frame_slice=...)` |
| VIPIR analysis | 2.0 documents `PolarizationClassifier` and `IonogramScaler` |
| Docs | Versioned builds (1.0 to 2.0) on Read the Docs, with this page and a changelog |
| Tooling | `make env / install-dev / test / format / docs`, run in the `pynasonde` conda env |

## VIPIR analysis scope

The other `pynasonde.vipir.analysis` modules (`absorption`, `inversion`,
`spread_f`, `irregularities`, `nextyz`, `es_imaging`) are unchanged and still
importable, but they are not part of the 2.0 documented API yet. Their pages
remain in the [1.3 documentation](https://pynasonde.readthedocs.io/en/v1.3.0/).

## New examples and API pages

- [O/X mode separation](../examples/vipir/polarization.md) and
  [automatic scaling](../examples/vipir/scaler.md) for VIPIR.
- [RSF Viterbi autoscaling](../examples/digisonde/rsf_scaler.md) and the
  [Digisonde autoscaler API](../dev/digisonde/scaler/index.md).
