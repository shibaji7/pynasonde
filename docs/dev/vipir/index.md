# VIPIR (pynasonde.vipir)

This section documents the VIPIR-related modules in the `pynasonde.vipir` package.

Contents

- `ngi/` — NGI ionogram reader, autoscaler, and plotting utilities
- `riq/parsers/` — RIQ binary reader, IQ-data phase/velocity extraction
- `riq/datatypes/` — system and pulse configuration table dataclasses

Each page combines a short narrative, mkdocstrings directives to auto-generate API docs, and small usage examples.

Example quick start

```py
from pynasonde.vipir.ngi.source import DataSource

ds = DataSource(source_folder="path/to/ngi/", file_ext="*.ngi.bz2")
print(ds.file_names)
```
