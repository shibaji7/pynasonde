# VIPIR (pynasonde.vipir)

This section documents the VIPIR-related modules in the `pynasonde.vipir` package.

Contents

- `ngi/` — NGI ionogram reader, autoscaler, and plotting utilities
- `riq/parsers/` — RIQ binary reader, IQ-data phase/velocity extraction
- `riq/datatypes/` — system and pulse configuration table dataclasses

Each page combines a short narrative, mkdocstrings directives to auto-generate API docs, and small usage examples.

Example quick start

```py
# Short package-level imports (all forms are equivalent):

# Option A — via pynasonde.vipir
from pynasonde.vipir import DataSource, RiqDataset, Echo, EchoExtractor

# Option B — top-level package
from pynasonde import DataSource, RiqDataset, Echo, EchoExtractor

# Option C — original deep import (still valid)
from pynasonde.vipir.ngi.source import DataSource
from pynasonde.vipir.riq.parsers.read_riq import RiqDataset
from pynasonde.vipir.riq.echo import Echo, EchoExtractor

# Load NGI scaled ionogram files
ds = DataSource(source_folder="path/to/ngi/", file_ext="*.ngi.bz2")
print(ds.file_names)

# Extract Dynasonde-style seven-parameter echoes from a RIQ file
dataset = RiqDataset.create_from_file("path/to/file.RIQ")
extractor = EchoExtractor(dataset.sct, dataset.pulsets).extract()
df = extractor.to_dataframe()   # columns: height_km, xl_km, yl_km, velocity_mps, …
```
