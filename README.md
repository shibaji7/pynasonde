
# Pynasonde
<div style="text-align: center;">
  <img src="docs/assets/Colab-pynasonde-logo2.jpg" alt="Pynasonde" width="50%">
</div>

[![License: MIT](https://img.shields.io/badge/License%3A-MIT-green)](https://choosealicense.com/licenses/mit/) 
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/release/python-3110/) 
![GitHub Stable Release (latest by date)](https://img.shields.io/github/v/release/shibaji7/pynasonde)
[![Documentation Status](https://img.shields.io/readthedocs/pynasonde?logo=readthedocs&label=docs)](https://pynasonde.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/shibaji7/pynasonde/branch/main/graph/badge.svg)](https://codecov.io/gh/shibaji7/pynasonde)
[![Roadmap](https://img.shields.io/badge/roadmap-published-blue)](ROADMAP.md)

**[📖 Documentation](https://pynasonde.readthedocs.io/) · [📦 PyPI](https://pypi.org/project/pynasonde/) · [🐙 GitHub](https://github.com/shibaji7/pynasonde) · [🐛 Issues](https://github.com/shibaji7/pynasonde/issues) · [📜 Changelog](CHANGELOG.md) · [🗺️ Roadmap](ROADMAP.md) · [🤝 Contributing](CONTRIBUTING.md)**


Pynasonde is an open-source Python-based application designed for precision ionospheric radio sounding, with a strong focus on analyzing the phase characteristics of radio echoes. Tailored for Space Weather applications, Pynasonde offers a suite of unique tools that help extract valuable insights directly and autonomously from ionogram data.

With Pynasonde, you can achieve accurate echo recognition, noise discrimination, and echo classification into traces. The application also facilitates the scaling of standard ionospheric parameters, 3-D plasma density inversion (including error bars), diagnostics of small-scale irregularities, and determination of vector velocities.

Designed as a comprehensive toolbox, Pynasonde empowers researchers to process raw ionosonde datasets efficiently, providing reliable, real-time insights into ionospheric conditions and phenomena. Whether you're focused on space weather forecasting, radio communication, or scientific exploration, Pynasonde is your go-to tool for precision ionospheric analysis.

## Installation

```bash
pip install pynasonde              # lean core
pip install "pynasonde[digisonde]" # Digisonde parsers and plots
pip install "pynasonde[vipir]"     # VIPIR NGI/RIQ tools
pip install "pynasonde[all]"       # everything
```

See the [install guide](https://pynasonde.readthedocs.io/en/latest/user/install/) for all extras,
[What's new in 2.0](https://pynasonde.readthedocs.io/en/latest/user/whats_new_2_0/) if you are upgrading
from 1.x, and the [CHANGELOG](CHANGELOG.md). Docs for each release are on Read the Docs
(use the version menu).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup (`make env`,
`make test`, `make format`), branch model, commit conventions, test and
documentation conventions, the pull-request checklist, and the release procedure.
See [ROADMAP.md](ROADMAP.md) for what is done, in progress, and open for
contribution, and [CHANGELOG.md](CHANGELOG.md) for what changed in each release.

## Source Code 

The library source code can be found on the [pynasonde GitHub](https://github.com/shibaji7/pynasonde) repository. 

If you have any questions or concerns please submit an **Issue** on the [pynasonde GitHub](https://github.com/shibaji7/pynasonde) repository. 

## Documentation
Read the docs: https://pynasonde.readthedocs.io/en/latest/