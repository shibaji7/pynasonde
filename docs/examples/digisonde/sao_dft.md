# SAO Isodensity Contours + DFT Doppler Waterfall and Spectra

<div class="hero">
  <h3>Three-Figure Workflow: Isodensity, Waterfall, and Spectra</h3>
  <p>
    Build a daily isodensity contour from 240 SAO files, then visualize the
    Doppler waterfall and per-height spectra from a single DFT drift file —
    all three plots in one script.
  </p>
</div>

This page explains `examples/digisonde/sao_iso_dft_plots.py`.

Data used: KR835 (Kirtland AFB), 14 October 2023 —
240 `.SAO` files + `KR835_2023287000915.DFT`.

## Call Flow

### Figure 1 — Isodensity contour

1. `SaoExtractor.load_SAO_files(...)` loads all 240 SAO files in parallel.
2. `df["datetime"]` is cast to `pd.Timestamp` for the time axis.
3. `SaoSummaryPlots.add_isodensity_contours(...)` bins height to a 5 km grid
   with `pd.cut` + `groupby`, renders a `pcolormesh` colored by mean plasma
   frequency, and overlays `contour` lines at each integer MHz level
   (mimicking [Digisonde-Isodensity.gif](https://digisonde.com/images/Digisonde-Isodensity.gif)).

### Figure 2 — Doppler waterfall

1. `DftExtractor(filepath, ...)` opens the DFT file and counts blocks.
2. `.extract()` iterates all 96 blocks, unpacking 16 sub-cases × 128
   amplitude bytes + 128 phase bytes per sub-case.  Header bits are
   decoded from the LSBs of all amplitude bytes.
3. `.to_pandas()` flattens to rows of `(block_idx, subcase_idx, height_km,
   doppler_bin, amplitude, phase, frequency_hz, date)`.
4. `SkySummaryPlots.plot_doppler_waterfall(...)` auto-selects the block with
   peak amplitude, computes the 2nd–98th percentile color range, and renders
   a `pcolormesh` of Doppler bin vs. height.

### Figure 3 — Doppler spectra

1. Same DataFrame from step 3 above.
2. `SkySummaryPlots.plot_doppler_spectra(...)` auto-selects the same best
   block, samples `n_heights` evenly-spaced height bins, and plots one
   amplitude-vs-Doppler line per height, colored by viridis.

## Key Code

### 1) SAO Isodensity Contours

```python
import pandas as pd
from pynasonde.digisonde.parsers.sao import SaoExtractor
from pynasonde.digisonde.digi_plots import SaoSummaryPlots

SAO_DIR = "path/to/SKYWAVE_DPS4D_2023_10_14"

df_sao = SaoExtractor.load_SAO_files(
    folders=[SAO_DIR],
    ext="KR835_*.SAO",
    n_procs=8,
    func_name="height_profile",
)
df_sao = df_sao.reset_index(drop=True)
df_sao["datetime"] = pd.to_datetime(df_sao["datetime"])

p = SaoSummaryPlots(figsize=(10, 4), font_size=10)
p.add_isodensity_contours(
    df_sao,
    xparam="datetime",
    yparam="th",
    zparam="pf",
    ylim=[50, 500],
    fbins=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    text="KR835  2023-10-14",
)
p.save("docs/examples/figures/sao_isodensity_KR835.png")
p.close()
```

### 2) DFT Doppler Waterfall

```python
from pynasonde.digisonde.parsers.dft import DftExtractor
from pynasonde.digisonde.digi_plots import SkySummaryPlots

DFT_FILE = f"{SAO_DIR}/KR835_2023287000915.DFT"

dft = DftExtractor(DFT_FILE, extract_time_from_name=True, extract_stn_from_name=True)
dft.extract()
df_dft = dft.to_pandas()
title_dft = f"KR835  {dft.date.strftime('%Y-%m-%d  %H:%M:%S')} UT"

sk = SkySummaryPlots(figsize=(7, 5), font_size=10, subplot_kw={})
sk.plot_doppler_waterfall(
    df_dft,
    cmap="inferno",
    text=title_dft,
)
sk.save("docs/examples/figures/dft_doppler_waterfall_KR835.png")
sk.close()
```

### 3) DFT Doppler Spectra

```python
sk2 = SkySummaryPlots(figsize=(7, 4), font_size=10, subplot_kw={})
sk2.plot_doppler_spectra(
    df_dft,
    n_heights=8,
    cmap="viridis",
    text=title_dft,
)
sk2.save("docs/examples/figures/dft_doppler_spectra_KR835.png")
sk2.close()
```

!!! note "subplot_kw override"
    `SkySummaryPlots` defaults to a polar projection for skymap use.
    Always pass `subplot_kw={}` when creating waterfall or spectra plots.

## DFT Height Decoding

Virtual height is estimated from the block header fields:

```
height_km = 80 + subcase_idx × height_resolution × 5
```

where `height_resolution` is the raw 4-bit header value (typical value = 2,
giving 10 km steps).  For the KR835 file this yields heights 100–250 km —
physically reasonable for E/F-layer sounding at ~6.5 MHz.

## Run

```bash
cd /home/chakras4/Research/CodeBase/pynasonde
python examples/digisonde/sao_iso_dft_plots.py
```

## Output Figures

<figure markdown>
![SAO Isodensity Contours](../figures/sao_isodensity_KR835.png)
<figcaption>Figure 1: Daily isodensity contour for KR835, 14 October 2023. Time on the x-axis, virtual height on the y-axis; color indicates plasma frequency (MHz) at each height.</figcaption>
</figure>

<figure markdown>
![DFT Doppler Waterfall](../figures/dft_doppler_waterfall_KR835.png)
<figcaption>Figure 2: Doppler waterfall from <code>KR835_2023287000915.DFT</code>. Doppler bin on the x-axis, height on the y-axis; amplitude color highlights the dominant drift signal.</figcaption>
</figure>

<figure markdown>
![DFT Doppler Spectra](../figures/dft_doppler_spectra_KR835.png)
<figcaption>Figure 3: Per-height Doppler spectra for the same DFT block. Each line corresponds to a sampled height bin, colored from low (purple) to high (yellow) altitude.</figcaption>
</figure>

## Related Files

- `examples/digisonde/sao_iso_dft_plots.py`
- `pynasonde/digisonde/parsers/sao.py`
- `pynasonde/digisonde/parsers/dft.py`
- `pynasonde/digisonde/digi_plots.py` — `SaoSummaryPlots.add_isodensity_contours()`,
  `SkySummaryPlots.plot_doppler_waterfall()`, `SkySummaryPlots.plot_doppler_spectra()`

## See Also

- [SAO Height Profiles and F2 Diagnostics](sao.md)
- [DIGISONDE DFT Format Guide](../../user/digisonde.md#dft----doppler-fourier-spectra)
- [DFT API Reference](../../dev/digisonde/parsers/dft.md)
