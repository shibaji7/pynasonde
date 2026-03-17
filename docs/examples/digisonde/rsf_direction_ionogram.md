# RSF Direction Ionogram and Daily Directogram

<div class="hero">
  <h3>Echo Direction Visualization — Figure 3-8 and Figure 3-12 Style</h3>
  <p>
    Produce a direction-coded ionogram for a single RSF sounding (mimicking
    Digisonde-4D Manual Figure 3-8) and a full-day directogram stacking all
    soundings by UT time with West–East ground distance on the x-axis
    (Figure 3-11/3-12 style).
  </p>
</div>

This page explains `examples/digisonde/rsf_direction_ionogram.py`.

Data used: KR835 (Kirtland AFB), 14 October 2023, 480 RSF files.

## Call Flow

### Single direction-coded ionogram

1. `RsfExtractor(filepath, ...)` parses one `.RSF` file.
2. `.extract()` → `.to_pandas()` flattens echoes to a DataFrame with columns
   `frequency`, `height_km`, `amplitude`, `azimuth`, `doppler_num`, `polarization`.
3. `RsfIonogram.add_direction_ionogram(...)` classifies each echo into one of
   10 direction+polarization categories (NoVal, NNE, E, W, Vo−, Vo+, SSW, X−, X+,
   NNW) and scatter-plots (log₁₀ frequency, height) with per-category colors
   matching Figure 3-8.
4. Legend uses `matplotlib.patches.Patch` entries for each active category.

### Daily directogram

1. `glob.glob("KR835_*.RSF")` collects all 480 RSF files for the day.
2. Each file is parsed with `RsfExtractor`; resulting DataFrames are concatenated.
3. `RsfIonogram.add_directogram(...)` groups echoes by ionogram timestamp, computes
   the vertical reference height H_v from peak vertical echoes, and derives
   ground distance D_i = √(H_i² − H_v²) with sign negative for westward arrivals.
4. Y-axis = UT time (`matplotlib.dates`), X-axis = D_i (km), scatter colored by
   echo category.

## Key Code

### 1) Single Ionogram — Direction-Coded

```python
from pynasonde.digisonde.parsers.rsf import RsfExtractor
from pynasonde.digisonde.digi_plots import RsfIonogram

RSF_FILE = "path/to/KR835_2023287000000.RSF"

extractor = RsfExtractor(RSF_FILE, extract_time_from_name=True, extract_stn_from_name=True)
extractor.extract()
df = extractor.to_pandas()

h = extractor.rsf_data.rsf_data_units[0].header
title = f"KR835  {h.date.strftime('%Y-%m-%d  %H:%M:%S')} UT"

r = RsfIonogram(figsize=(6, 5), font_size=10)
r.add_direction_ionogram(
    df,
    ylim=[80, 600],
    xlim=[1, 15],
    xticks=[1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0],
    text=title,
    lower_plimit=5,
    ms=1.0,
)
r.save("tmp/rsf_direction_ionogram_KR835.png")
r.close()
```

### 2) Daily Directogram

```python
import glob
import pandas as pd
from pynasonde.digisonde.parsers.rsf import RsfExtractor
from pynasonde.digisonde.digi_plots import RsfIonogram

RSF_DIR = "path/to/SKYWAVE_DPS4D_2023_10_14"

frames = []
for fpath in sorted(glob.glob(f"{RSF_DIR}/KR835_*.RSF")):
    try:
        ex = RsfExtractor(fpath, extract_time_from_name=True)
        ex.extract()
        frames.append(ex.to_pandas())
    except Exception as e:
        print(f"Skipped {fpath}: {e}")

df_day = pd.concat(frames, ignore_index=True)

r = RsfIonogram(figsize=(6, 8), font_size=10)
r.add_directogram(
    df_day,
    dlim=[-800, 800],
    lower_plimit=5,
    ms=0.5,
    text="KR835  2023-10-14",
)
r.save("tmp/rsf_directogram_KR835_daily.png")
r.close()
```

## Echo Direction Color Scheme

| Category | Color | Azimuth | Notes |
|----------|-------|---------|-------|
| NoVal | gray | — | Below amplitude threshold |
| NNE | royalblue | 0° | North / north-northeast |
| E | dodgerblue | 60° | East |
| W | gold | 120° | Southeast (maps west in D_i) |
| Vo− | darkred | vertical | Negative Doppler (downward layer) |
| Vo+ | lightcoral | vertical | Positive Doppler (upward layer) |
| SSW | orange | 180° | South / south-southwest |
| X− | darkgreen | — | X-mode, negative Doppler |
| X+ | lightgreen | — | X-mode, positive Doppler |
| NNW | midnightblue | 300° | North-northwest |

## Ground Distance Formula

For each echo at virtual height H_i, ground distance is:

```
D_i = sqrt(H_i² − H_v²)
```

where H_v is the ionogram's vertical reference height (median of vertical
echoes near the F-layer peak).  Sign: negative for westward arrivals
(SSW, NNW, W), positive for eastward (NNE, E).

## Run

```bash
cd /home/chakras4/Research/CodeBase/pynasonde
python examples/digisonde/rsf_direction_ionogram.py
```

## Output Figures

- `tmp/rsf_direction_ionogram_KR835.png` — single direction-coded ionogram
- `tmp/rsf_directogram_KR835_daily.png` — full-day directogram

## Related Files

- `examples/digisonde/rsf_direction_ionogram.py`
- `pynasonde/digisonde/parsers/rsf.py`
- `pynasonde/digisonde/digi_plots.py` — `RsfIonogram.add_direction_ionogram()`, `add_directogram()`

## See Also

- [RSF Parse and Inspect](rsf.md)
- [DIGISONDE RSF Format Guide](../../user/digisonde.md#rsf----raw-sounding-file)
- [RSF API Reference](../../dev/digisonde/parsers/rsf.md)
