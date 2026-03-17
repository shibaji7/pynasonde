# Raw IQ Processing (pynasonde.digisonde.raw)

This section documents the raw IQ processing pipeline for DPS4D Digisonde recordings.

Contents

- `raw_parse.md` — `process()` function: full sounding pipeline (IQ → complementary-code correlation → ionogram NetCDF)
- `iq_reader.md` — `IQStream` reader: time-partitioned binary file management
- `raw_plots.md` — `RawPlots` / `AFRLPlots`: PSD and spectrogram visualisation helpers

## Example quick start

```py
from pynasonde.digisonde.raw.raw_parse import process
import datetime as dt

program = {
    "Epoch": dt.datetime(2023, 10, 14, 16, 0, tzinfo=dt.timezone.utc),
    "ID": "DPS4D_Kirtland0",
    "FFTMode": False,
    "rxTag": "ch0",
    "Save Phase": False,
    "Freq Stepping Law": "linear",
    "Lower Freq Limit": 2e6,
    "Upper Freq Limit": 15e6,
    "Coarse Freq Step": 30e3,
    "Number of Fine Steps": 1,
    "Fine Freq step": 5e3,
    "Fine Multiplexing": False,
    "Inter-Pulse Period": 10e-3,
    "Number of Integrated Repeats": 8,
    "Interpulse Phase Switching": False,
    "Wave Form": "16-chip complementary",
    "Polarization": "O and X",
}
process(program, dir_iq="/media/chakras4/69F9D939661D263B", out_dir="out/")
```

::: pynasonde.digisonde.raw.raw_parse
    handler: python
    options:
        show_root_heading: true
        show_source: false
        members:
            - process
