<!--
Author(s): Shibaji Chakraborty

Disclaimer:
-->

# Installing Pynasonde

<div class="hero">
  <h2>Get Up and Running in Minutes</h2>
  <p>
    Pynasonde requires Python 3.11+ and installs cleanly via pip into any
    virtual environment. Most users only need the one-line install below.
  </p>
</div>

[![License: MIT](https://img.shields.io/badge/License%3A-MIT-green)](https://choosealicense.com/licenses/mit/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
![GitHub Stable Release (latest by date)](https://img.shields.io/github/v/release/shibaji7/pynasonde)

## Quick Start

```bash
pip install pynasonde            # lean core
pip install "pynasonde[all]"     # everything (what 1.x used to install)
```

Verify the install:

```python
import pynasonde
print(pynasonde.__version__)
```

!!! note "Upgrade"
    Already installed? Run `pip install --upgrade pynasonde`. Coming from 1.x?
    The 2.0 core install is lean, so read [What's new in 2.0](whats_new_2_0.md).

!!! tip "Pick another version of the docs"
    This page documents the version selected in the version menu. Installing an
    older release: `pip install "pynasonde==1.3.0"` (1.x installs all dependencies).

## Choose What to Install

The core install has only `numpy`, `pandas`, `matplotlib`, `loguru`, `toml` and
`tqdm`. Add extras for the modules you use:

| Extra | Adds | Use it for |
|-------|------|------------|
| `digisonde` | `lxml`, `scipy`, `xarray`, `netCDF4`, `pytz`, `timezonefinder`, `joblib` | SAO, RSF, SBF, MMM, DFT, DVL, SKY, EDP, CADI and raw IQ parsers and plots |
| `digisonde-scaler` | `digisonde` + `scikit-learn` | Digisonde autoscaler (`pynasonde.digisonde.scaler`) |
| `digisonde-db` | `digisonde` + `fdb` | DIDBase (Firebird) connector |
| `ocr` | `opencv-python-headless`, `pytesseract` | Ionogram image parser (also needs the `tesseract` system package) |
| `vipir` | `scipy`, `xarray`, `netCDF4`, `pytz`, `timezonefinder`, `pysolar`, `joblib`, `scikit-learn`, `scikit-image`, `opencv-python-headless` | VIPIR NGI and RIQ readers and tools |
| `vipir-analysis` | nothing extra | `PolarizationClassifier` and `IonogramScaler` (core only; reading RIQ files needs `vipir`) |
| `time` | `pytz`, `timezonefinder` | Local-time conversion (`TimeZoneConversion`) |
| `web` | `requests`, `beautifulsoup4`, `lxml` | `Webhook` NGI/RIQ downloads |
| `plots` | `SciencePlots` | Optional publication plot styles |
| `all` | all of the above except `matlab` | Full 1.x-equivalent install |
| `dev` / `docs` | test, format and docs tooling | Contributors |

Combine extras with commas, for example:

```bash
pip install "pynasonde[digisonde,vipir]"
```

If you use a class without its extra installed, Python raises a
`ModuleNotFoundError` naming the missing package; install the matching extra.

## Recommended Workflows

### conda (recommended)

```bash
conda create -n pynasonde python=3.11
conda activate pynasonde
pip install pynasonde
```

### pip virtual environment

```bash
python3 -m venv pynasonde-env
source pynasonde-env/bin/activate   # Windows: pynasonde-env\Scripts\activate
pip install pynasonde
```

## Developer Install

Clone the repository and install in editable mode so local source changes take effect immediately:

```bash
git clone https://github.com/shibaji7/pynasonde.git
cd pynasonde
pip install -e ".[dev]"
```

The `[dev]` extra pulls in `pytest`, `pytest-cov`, `coverage`, `black`, `isort`,
`autoflake`, `build` and every module extra, so the whole test suite can run.

The repository's `Makefile` wraps the common tasks and runs them in a conda
environment named `pynasonde` (`make env` creates it from `env.yml`):

```bash
make env          # create/update the conda dev environment
make install-dev  # editable install with dev extras
make test         # full test suite with coverage
make format       # autoflake + isort + black
make docs         # build the documentation site
```

!!! warning "Cartopy (optional)"
    Coastline and geographic projection plots require `cartopy >= 0.19`.
    Install it separately following the [Cartopy installation guide](https://scitools.org.uk/cartopy/docs/latest/installing.html).
    On Ubuntu: `sudo apt-get install python3-cartopy` often works cleanly.

## Core Dependencies

Installed automatically with `pip install pynasonde` (minimum versions, taken from `pyproject.toml`):

| Package | Minimum | Purpose |
|---------|---------|---------|
| `numpy` | 1.26 | Array operations, bit-level unpacking |
| `pandas` | 2.2 | Tidy DataFrames for all parsed outputs |
| `matplotlib` | 3.9 | Plotting |
| `loguru` | any | Structured logging |
| `toml` | any | Configuration files |
| `tqdm` | any | Progress bars |

## System Requirements

| OS | Required system package |
|----|------------------------|
| Ubuntu / Debian | `libyaml-dev` |
| OpenSuse | `python3-PyYAML` |
| Fedora | `libyaml-devel` |
| macOS | Xcode Command Line Tools |
| Windows | pip (no extra system package) |

Check your Python version:

```bash
python --version   # or python3 --version
```
