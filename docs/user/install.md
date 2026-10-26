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
pip install "pynasonde[all]"     # everything
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
    older release: `pip install "pynasonde==1.3.0"`. Note that the published 1.x packages
    declare no dependencies, so install numpy, pandas, matplotlib and the rest yourself
    (or use a conda environment that already has them).

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
| `all` | every module extra above (not `matlab`) | Everything at once |
| `dev` / `docs` | test, format and docs tooling | Contributors |

Combine extras with commas, for example:

```bash
pip install "pynasonde[digisonde,vipir]"
```

## How lazy installs behave

pynasonde imports its modules **lazily**. `import pynasonde` is fast and loads none of the
optional packages; a module is imported the first time you use one of its names. So:

- Installing only some extras is fine. Everything that does not need a missing package keeps working.
- If you use something whose extra is not installed, Python raises a
  `ModuleNotFoundError` that names the missing package, for example
  `No module named 'lxml'` when you use `SaoExtractor` on a core-only install. The tables
  below turn that package name into the extra to install.
- Existing 1.x imports such as `from pynasonde import SaoExtractor` still work; they just
  need the extra their module belongs to.

Check what you have:

```bash
python -c "import pynasonde; print(pynasonde.__version__)"
python -c "from pynasonde.digisonde import SaoExtractor"   # needs the digisonde extra
python -c "from pynasonde.vipir import RiqDataset"         # needs the vipir extra
```

## Which extra do I need?

Measured by importing each name on an install that has only that extra:

| You want to use | Install |
|-----------------|---------|
| `PolarizationClassifier`, `IonogramScaler` (`pynasonde.vipir.analysis`) | core only |
| Scaler utilities: `normalize_echo_dataframe`, `detect_echo_schema`, `classify_solar_regime`, `solar_zenith_angle`, `get_viterbi_preset`, `ScaleHmF2`, `despike_scaled_parameters` (`pynasonde.digisonde.scaler`) | core only |
| `IonogramAutoScaler`, `ViterbiTracer`, the RFI filters, `scale_dataframe`, `scale_dataframes`, `scale_dataframe_with_preset` | `digisonde-scaler` |
| `SaoExtractor`, `RsfExtractor`, `SbfExtractor`, `DftExtractor`, `DvlExtractor`, `ModMaxExtractor`, `SkyExtractor`, `EdpExtractor`, `CadiReader`, `CadiExtractor`, and the plot classes (`SaoSummaryPlots`, `SkySummaryPlots`, `RsfIonogram`, `CadiIonogram`, `DigiPlots`) | `digisonde` |
| `IonogramImageExtractor` | `ocr` |
| `DidBaseConnector` (`pynasonde.digisonde.didbase`; importing is light, connecting needs `fdb`) | `digisonde-db` |
| `DataSource`, `Dataset`, `Trace`, `AutoScaler`, `Ionogram`, `NoiseProfile`, `RiqDataset`, `Echo`, `EchoExtractor`, `IonogramFilter` | `vipir` |
| `Webhook` | `web` |
| `TimeZoneConversion` (importing it is light; creating an instance needs `pytz` and `timezonefinder`) | `time` |

## Missing module? Install this

| Error says `No module named ...` | Install |
|----------------------------------|---------|
| `lxml`, `scipy`, `xarray`, `netCDF4`, `pytz`, `timezonefinder`, `joblib` | `digisonde` (or `vipir` for the VIPIR tools) |
| `sklearn` (scikit-learn) | `digisonde-scaler` or `vipir` |
| `cv2` (OpenCV), `pytesseract` | `ocr` (`cv2` is also part of `vipir`) |
| `skimage`, `pysolar` | `vipir` |
| `requests`, `bs4` (BeautifulSoup) | `web` |
| `fdb` | `digisonde-db` |
| `scienceplots` | `plots` |
| `matlab` | the MATLAB Engine API for Python (not on PyPI) |

!!! note "pytesseract needs a system program"
    `pytesseract` calls the `tesseract` program, which pip cannot install. On Ubuntu:
    `sudo apt-get install tesseract-ocr`; with conda: `conda install -c conda-forge tesseract`.

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
