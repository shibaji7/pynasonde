<!--
Author(s): Shibaji Chakraborty

Disclaimer:

-->

# Pynasonde

<div class="hero">
  <h2>Precision Ionospheric Radio Sounding in Python</h2>
  <p>
    Pynasonde is a Python toolkit for ingesting, analyzing, and visualizing
    ionosonde data from DIGISONDE DPS4D and VIPIR radar systems — built for
    Space Weather research and operational ionospheric monitoring.
  </p>
</div>

!!! warning "Beta Status"
    Pynasonde is in active development. APIs and documentation may change as features are added and validated.

<div style="text-align: center;">
  <img src="assets/Colab-pynasonde-logo2.jpg" alt="Pynasonde" width="40%">
</div>

[![License: MIT](https://img.shields.io/badge/License%3A-MIT-green)](https://choosealicense.com/licenses/mit/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
![GitHub Stable Release (latest by date)](https://img.shields.io/github/v/release/shibaji7/pynasonde)
[![Documentation Status](https://img.shields.io/readthedocs/pynasonde?logo=readthedocs&label=docs)](https://pynasonde.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/shibaji7/pynasonde/branch/main/graph/badge.svg)](https://codecov.io/gh/shibaji7/pynasonde)

Pynasonde reads raw ionosonde data files (`.RSF`, `.SBF`, `.DFT`, `.SAO`, `.SKY`, `.DVL`,
`.RIQ`) and exposes them as tidy pandas DataFrames.  It ships ready-made
summary plotters for electron-density profiles, isodensity contours,
direction-coded ionograms, Doppler waterfalls, and plasma-drift directograms.

## Quick Start

```bash
pip install pynasonde
```

```python
from pynasonde.digisonde.parsers.sao import SaoExtractor

df = SaoExtractor.load_SAO_files(folders=["path/to/SAO/"], func_name="height_profile", n_procs=4)
print(df.head())
```

## Source Code

The library source code can be found on the [pynasonde GitHub](https://github.com/shibaji7/pynasonde) repository.

If you have any questions or concerns please submit an **Issue** on the [pynasonde GitHub](https://github.com/shibaji7/pynasonde) repository.

## Documentation Links

<div class="doc-card-grid">
  <div class="doc-card">
    <strong>Installation</strong>
    Setup guidance, virtual environments, and developer install.<br>
    <a href="user/install/">Open Installation</a>
  </div>
  <div class="doc-card">
    <strong>DIGISONDE</strong>
    DPS4D file formats: RSF, SBF, DFT, SAO, SKY, DVL — parsers and plotters.<br>
    <a href="user/digisonde/">Open DIGISONDE Guide</a>
  </div>
  <div class="doc-card">
    <strong>VIPIR</strong>
    RIQ file format, SCT/PCT structures, and ionogram analysis.<br>
    <a href="user/vipir/">Open VIPIR Guide</a>
  </div>
  <div class="doc-card">
    <strong>Examples</strong>
    End-to-end worked examples for SAO, RSF, DFT, and VIPIR data.<br>
    <a href="examples/digisonde/sao/">Open Examples</a>
  </div>
  <div class="doc-card">
    <strong>API Reference</strong>
    Module, class, and method reference pages.<br>
    <a href="dev/">Open API</a>
  </div>
  <div class="doc-card">
    <strong>Citing & Authors</strong>
    Citation guidance and contributor listing.<br>
    <a href="user/citing/">Citing</a> | <a href="user/authors/">Authors</a>
  </div>
</div>
