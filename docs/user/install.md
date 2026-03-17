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
pip install pynasonde
```

Verify the install:

```python
import pynasonde
print(pynasonde.__version__)
```

!!! note "Upgrade"
    Already installed? Run `pip install --upgrade pynasonde` to get the latest release.

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

The `[dev]` extra pulls in `pytest`, `pytest-cov`, `coverage`, and the full
scientific stack (`numpy`, `pandas`, `matplotlib`, `scipy`, …).

!!! warning "Cartopy (optional)"
    Coastline and geographic projection plots require `cartopy >= 0.19`.
    Install it separately following the [Cartopy installation guide](https://scitools.org.uk/cartopy/docs/latest/installing.html).
    On Ubuntu: `sudo apt-get install python3-cartopy` often works cleanly.

## Core Dependencies

Pynasonde's `setup.py` installs these automatically:

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | 1.26.4 | Array operations, bit-level unpacking |
| `pandas` | 2.2.3 | Tidy DataFrames for all parsed outputs |
| `matplotlib` | 3.9.2 | Publication-quality plotting |
| `scipy` | 1.14.1 | Signal processing utilities |
| `xarray` | 2024.9.0 | Gridded data structures |
| `loguru` | latest | Structured logging |
| `timezonefinder` | 6.5.5 | Station local-time lookup |
| `SciencePlots` | 2.1.1 | Clean scientific plot styles |

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
