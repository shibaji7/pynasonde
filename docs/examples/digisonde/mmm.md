# MMM — ModMax Ionogram Viewer

<div class="hero">
  <h3>End-to-End MMM Workflow</h3>
  <p>
    Parse a DPS4D <code>.MMM</code> (ModMax) binary file, decode O/X
    polarisation and Doppler channels, and produce publication-ready
    pcolormesh ionogram figures.
  </p>
</div>

This page explains `examples/digisonde/mmm.py`.

Data used: AU930 (Wallops Island), DOY 147, 2017 — a single sounding block.

## Background

MMM files store maximum-amplitude echoes across all Doppler channels. Each
range-frequency bin encodes:

- **bits 3**: polarisation — `0 = O-mode`, `1 = X-mode`
- **bits 2–0**: Doppler channel index (0–7)

The brightest-echo convention (max amplitude across Doppler bins) matches the
SAOExplorer display.

## Call Flow

1. `ModMaxExtractor(file, extract_time_from_name=True)` opens the binary and
   decodes the header.
2. `.extract()` iterates blocks and populates a list of records.
3. `.to_pandas()` returns a flat DataFrame with columns
   `frequency_mhz`, `range_km`, `amplitude_dB`, `polarization`, `doppler_channel`, `datetime`.
4. Rows below a noise floor are dropped and the data is pivoted to a
   height × frequency grid with `.pivot_table()`.
5. Two figures are saved: combined O+X and side-by-side O/X mode.

## Key Code

### 1) Load and Parse

```python
from pynasonde.digisonde.parsers.mmm import ModMaxExtractor
from pynasonde.digisonde.digi_utils import setsize

setsize(14)

ext = ModMaxExtractor(
    "AU930_2017147000005.MMM",
    extract_time_from_name=True,
    extract_stn_from_name=True,
)
ext.extract()
df = ext.to_pandas()
```

### 2) Noise Filter and Grid

```python
import numpy as np

NOISE_FLOOR = 10   # dB
VMIN, VMAX  = 10, 84

df = df[df["amplitude_dB"] > NOISE_FLOOR]

freq_bins   = np.sort(df["frequency_mhz"].unique())
height_bins = np.sort(df["range_km"].unique())
F, H = np.meshgrid(freq_bins, height_bins)

def _to_grid(sub):
    return (
        sub.pivot_table(
            index="range_km", columns="frequency_mhz",
            values="amplitude_dB", aggfunc="max",
        )
        .reindex(index=height_bins, columns=freq_bins)
        .values
    )
```

### 3) Combined O+X Ionogram

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))
pc = ax.pcolormesh(F, H, _to_grid(df),
                   cmap="plasma", vmin=VMIN, vmax=VMAX, shading="nearest")
ax.set_xscale("log")
ax.set_xlim(0.9, 14)
ax.set_ylim(50, 600)
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Virtual Height (km)")
ax.set_title("MMM Ionogram (O+X)")
plt.colorbar(pc, ax=ax, label="Amplitude (dB)")
fig.savefig("docs/examples/figures/mmm_ionogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

### 4) Side-by-Side O / X Mode

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, (pol, cmap) in zip(axes, [("O", "plasma"), ("X", "viridis")]):
    sub = df[df["polarization"] == pol]
    pc  = ax.pcolormesh(F, H, _to_grid(sub),
                        cmap=cmap, vmin=VMIN, vmax=VMAX, shading="nearest")
    ax.set_xscale("log"); ax.set_xlim(0.9, 14); ax.set_ylim(50, 600)
    ax.set_title(f"{pol}-mode ({len(sub):,} pts)")
    plt.colorbar(pc, ax=ax, label="Amplitude (dB)")
fig.savefig("docs/examples/figures/mmm_ionogram_OX.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

## Run

```bash
cd /path/to/pynasonde
python examples/digisonde/mmm.py
```

## Related Files

- `examples/digisonde/mmm.py`
- `pynasonde/digisonde/parsers/mmm.py`
- `pynasonde/digisonde/datatypes/mmmdatatypes.py`

## See Also

- [RSF Parse and Inspect](rsf.md)
- [MMM API Reference](../../dev/digisonde/parsers/mmm.md)
