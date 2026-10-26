# pynasonde roadmap

*Last updated: 2026-10-26.*

This is the single place to see **what is done, what is partial, what is planned,
and where help is wanted**. For release-by-release history see
[CHANGELOG.md](CHANGELOG.md); for how to contribute see
[CONTRIBUTING.md](CONTRIBUTING.md).

| Status | Meaning |
|--------|---------|
| ✅ **Done** | Implemented, tested and documented. |
| 🟡 **Partial** | Works, but something around it is unfinished (docs, tests, typing or one dependency). |
| 🔵 **Planned** | Agreed but not started. |

---

## Release train

pynasonde follows [Semantic Versioning](https://semver.org/). The version comes
from the git tag, and every tagged release has its own versioned documentation on
[Read the Docs](https://pynasonde.readthedocs.io/). Dates are given only for
shipped releases; later milestones are driven by scope, not by the calendar.

| Milestone | Scope highlight | Status |
|-----------|-----------------|--------|
| **v1.0.0** | Initial release (SoftwareX submission), 2025-10-29 | ✅ released |
| **v1.3.0** | CADI MD2/MD4 support, GRM parser, MMM parser rewrite, DIDBase connector, 2026-06-26 | ✅ released |
| **v2.0.0** | Selective update of 1.x with **full backward compatibility**: lean core plus per-module install extras, lazy imports, version from the git tag, versioned docs, Digisonde autoscaler, CI per extra, Google-style docstrings and type hints in the documented scope | 🟡 in progress, see the checklist below |
| **v2.x** | Bring the remaining VIPIR analysis modules into the documented API, one at a time, each with examples and tests | 🔵 planned |
| **Later** | Typed package (`py.typed`), Python 3.13, more bundled sample data | 🔵 planned |

### v2.0.0 release checklist

- [x] Lean core install and per-module extras (`digisonde`, `digisonde-scaler`,
      `digisonde-db`, `ocr`, `vipir`, `vipir-analysis`, `time`, `web`, `plots`,
      `all`, `docs`, `dev`)
- [x] Lazy package imports; every 1.x import path still works
- [x] Version single-sourced from the git tag
- [x] Changelog, "What's new in 2.0" and install pages; docs config for versioned builds
- [x] Digisonde autoscaler (`pynasonde.digisonde.scaler`) with API pages, example and tests
- [x] 2.0 VIPIR analysis scope: `polarization` and `scaler` documented, with examples
- [x] CI workflow: full suite on Python 3.11 and 3.12, one job per install extra,
      lint, strict docs build (each job's commands were reproduced locally)
- [x] Google-style docstrings and type hints in the documented scope
- [x] Developer tooling: `make` targets, conda dev environment, `.env` credentials
- [ ] Push `main` and confirm the CI workflow is green on GitHub
- [ ] Activate the v1.x tags on Read the Docs and confirm each one builds
- [ ] Update the versions in `CITATION.cff` and `codemeta.json`
- [ ] Tag `v2.0.0` (`make release VERSION=2.0.0`), publish to PyPI

---

## Status by area

### Digisonde

| Area | Status | Notes |
|------|--------|-------|
| SAO, RSF, SBF, MMM, DFT, DVL, SKY, EDP, GRM parsers | ✅ | Docs, examples and tests |
| Ionogram image (OCR) parser | ✅ | Needs the `ocr` extra and the `tesseract` system package |
| CADI MD2/MD4 (`digisonde.cadi`) | ✅ | Reader, extractor, plots |
| Raw IQ pipeline (`digisonde.raw`) | ✅ | |
| Autoscaler (`digisonde.scaler`) | ✅ | New in 2.0. Needs the `digisonde-scaler` extra (scikit-learn); tested on synthetic data and two bundled RSF files |
| DIDBase connector (`digisonde.didbase`) | ✅ | Needs `digisonde-db`; the live login test is opt-in (`make test-live`) |
| Plotting (`SaoSummaryPlots`, `RsfIonogram`, ...) | ✅ | Works on Matplotlib 3.11 |

### VIPIR

| Area | Status | Notes |
|------|--------|-------|
| NGI reader and AutoScaler (`vipir.ngi`) | ✅ | |
| RIQ reader, echo extractor, `IonogramFilter` (`vipir.riq`) | ✅ | |
| Analysis: `polarization`, `scaler` | ✅ | Documented 2.0 scope; run on the core install |
| Analysis: `absorption`, `inversion`, `spread_f`, `irregularities`, `nextyz`, `es_imaging` | 🟡 | Unchanged, tested and still importable, but not part of the 2.0 documented API; pages remain in the 1.3 docs |

### Platform

| Area | Status | Notes |
|------|--------|-------|
| Packaging and extras | ✅ | `pyproject.toml` only; wheel verified |
| Lazy imports | 🟡 | Done for `pynasonde`, `digisonde`, `digisonde.parsers`, `digisonde.scaler`, `vipir`, `vipir.analysis`; `vipir.ngi` and `vipir.riq` inits are still eager |
| Versioned docs (Read the Docs) | 🟡 | Config in place; v1.0.0 and v1.3.0 build locally with their own requirements; tags must be activated in the dashboard |
| CI | 🟡 | Workflow written; has not yet run on GitHub |
| Docstrings | 🟡 | Google style across the documented scope; the six out-of-scope analysis modules still use numpy style |
| Type hints | 🟡 | Public functions annotated where the types are clear; `mypy` baseline is several hundred issues |
| `py.typed` marker | 🔵 | Not shipped until the `mypy` baseline is reduced, so downstream type checkers are not given loose annotations |

---

## Known issues

These are documented, not yet fixed (fixing them changes behaviour):

- `vipir/ngi/scale.py::median_filter_fos_hms` is unused and raises `NameError`
  (it uses undefined names); `vipir.ngi.utils.smooth` is the working helper.
- `webhook.Webhook` checks the status of the directory-listing response, not of the
  file download, after fetching each file.
- A few older docstrings group parameter names (`cadi/extractor.py`, `riq/datatypes/pct.py`);
  griffe warns about them but the docs still build.
- Several event-specific examples need data that is not in the repository
  (for example `examples/digisonde/plot_foF2_hpF_ip_shock_events.py`) and have
  hardcoded local paths.
- `CITATION.cff` and `codemeta.json` still carry version `1.0.0`.

---

## Open for contribution

Good places to start, roughly from smallest to largest:

1. Fix the known issues above, each with a regression test.
2. Make the remaining examples portable (environment-variable data paths with a
   safe default, as in `examples/digisonde/rsf_direction_ionogram.py`) and add a
   smoke test for each.
3. Make `vipir.ngi` and `vipir.riq` package inits lazy.
4. Reduce the `mypy` baseline (`make typecheck`), then enable `py.typed`.
5. Bring one of the remaining analysis modules (`absorption`, `inversion`,
   `spread_f`, `irregularities`, `nextyz`) into the documented API: Google
   docstrings, type hints, an example with bundled data, a smoke test and API pages.
6. Add small bundled sample files for the RIQ/NGI and CADI examples so more
   examples run out of the box (keep each file small).
7. Add Python 3.13 to the CI matrix.

If you want to take one, open an issue first so work is not duplicated.
