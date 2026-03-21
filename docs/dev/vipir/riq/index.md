# VIPIR RIQ (pynasonde.vipir.riq)

This section documents the RIQ binary-format sub-package.

Contents

- `echo.py` — Dynasonde-style seven-parameter echo extractor (`Echo`, `EchoExtractor`)
- `parsers/` — low-level binary reader and IQ-data processing functions
- `datatypes/` — `SctType` and `PctType` configuration-table dataclasses
- `utils.md` — shared RIQ utility functions

Example quick start

```py
# Load a RIQ file and extract Dynasonde-style echoes
from pynasonde.vipir.riq import RiqDataset, Echo, EchoExtractor

dataset = RiqDataset.create_from_file("path/to/file.RIQ")

extractor = EchoExtractor(
    sct=dataset.sct,
    pulsets=dataset.pulsets,
    snr_threshold_db=3.0,
).extract()

df = extractor.to_dataframe()
# df columns: frequency_khz, height_km, amplitude_db, gross_phase_deg,
#             doppler_hz, velocity_mps, xl_km, yl_km,
#             polarization_deg, residual_deg, snr_db

ds = extractor.to_xarray()   # CF-convention Dataset

# Access SCT datatypes directly
from pynasonde.vipir.riq.datatypes import SctType, PctType
```
