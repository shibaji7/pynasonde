# Raw IQ Pipeline — `raw_parse`

<span class="api-badge api-package">P</span>
`pynasonde.digisonde.raw.raw_parse` — full sounding pipeline (IQ → correlation → ionogram) and the `IonogramResult` container.

---

## IonogramResult

<span class="api-badge api-class">C</span>
Dataclass that holds all arrays produced by a single Digisonde sounding.

::: pynasonde.digisonde.raw.raw_parse.IonogramResult
    handler: python
    options:
        show_root_heading: true
        show_source: false
        members:
            - frequency_mhz
            - power_total
            - power_db
            - to_xarray
            - to_netcdf

---

## Functions

<span class="api-badge api-method">M</span>
`process` — run the full sounding pipeline and return an `IonogramResult`.

::: pynasonde.digisonde.raw.raw_parse.process
    handler: python
    options:
        show_root_heading: true
        show_source: false

<span class="api-badge api-method">M</span>
`read_pulse` — read raw IQ for a single pulse without running the full pipeline.

::: pynasonde.digisonde.raw.raw_parse.read_pulse
    handler: python
    options:
        show_root_heading: true
        show_source: false

---

## Quick start

```python
import datetime as dt
from pynasonde.digisonde.raw.raw_parse import process

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

result = process(program, dir_iq="/media/chakras4/69F9D939661D263B", out_dir="out/")
if result is not None:
    print(result.frequency_mhz)          # (n_freqs,) array in MHz
    ds = result.to_xarray()              # xarray.Dataset
    result.to_netcdf("out/ionogram.nc")  # write to disk
```

## Accessing raw IQ for a single pulse

```python
from pynasonde.digisonde.raw.raw_parse import read_pulse

iq = read_pulse(
    program,
    dir_iq="/media/chakras4/69F9D939661D263B",
    coarse_index=10,   # 10th coarse frequency step
    pol_index=0,       # O-mode
    rep_index=0,       # first repeat
    comp_index=0,      # complementary code A
)
# iq is a np.ndarray of np.complex64, length = file_size (≈ sample rate)
```
