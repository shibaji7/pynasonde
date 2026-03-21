# Digisonde (pynasonde.digisonde)

This section documents the Digisonde-related modules in the `pynasonde.digisonde` package.

Contents
- `digi_utils.md` — helpers and IO utilities
- `digi_plots.md` — plotting helpers and summary plot classes
- `parsers/` — per-file parser documentation (skymap, SAO, SBF, MMM, DFT, RSF, EDP, DVL)
- `raw.md` — raw binary/stream handling and plotting utilities
- `datatypes.md` — datatype definitions used by parsers

Each page combines a short narrative, mkdocstrings directives to auto-generate API docs, and small usage examples.

Example quick start

```py
# Short package-level imports (available after the sub-package __init__.py
# files were populated — any of these forms work interchangeably):

# Option A — short import via pynasonde.digisonde
from pynasonde.digisonde import SaoExtractor, SaoSummaryPlots, load_station_csv

# Option B — top-level package import
from pynasonde import SaoExtractor, SaoSummaryPlots

# Option C — original deep import (still valid)
from pynasonde.digisonde.parsers.sao import SaoExtractor
from pynasonde.digisonde.digi_plots import SaoSummaryPlots
from pynasonde.digisonde.digi_utils import load_station_csv

stations = load_station_csv()
print(stations.head())

# Create a simple plot container
plotter = SaoSummaryPlots(fig_title='Example', nrows=1, ncols=1)
# feed plotter with a pre-built pandas DataFrame of Digisonde parameters
# plotter.add_TS(df)
```
