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
# Basic example: load station metadata and create a simple plot
from pynasonde.digisonde.digi_utils import load_station_csv
from pynasonde.digisonde.digi_plots import SaoSummaryPlots

stations = load_station_csv()
print(stations.head())

# Create a simple plot container
plotter = SaoSummaryPlots(fig_title='Example', nrows=1, ncols=1)
# feed plotter with pre-built pandas DataFrame of Digisonde parameters
# plotter.add_TS(df)
```
