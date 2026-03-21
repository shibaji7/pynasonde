# IonogramFilter (pynasonde.vipir.riq.parsers.filter)

<span class="api-badge api-package">P</span>
`pynasonde.vipir.riq.parsers.filter` — Coherent post-extraction echo filter
for VIPIR RIQ soundings.

## Overview

`IonogramFilter` takes one or more
:class:`~pynasonde.vipir.riq.echo.EchoExtractor` objects (one per sounding)
and applies a five-stage cascade of filters to the extracted echo cloud,
rejecting RFI, non-planar returns, multi-hop echoes, and isolated noise.
When multiple soundings are supplied the filter also enforces temporal
coherence: a (frequency, height) cell must be populated in at least
`temporal_min_soundings` consecutive soundings to be retained.

## Processing pipeline

```
EchoExtractor(s)  ─┐
                   │
                   ▼
         IonogramFilter.filter()
                   │
          ┌────────┴────────────────────────────┐
          │  Stage 1: RFI blanking              │  per-frequency height IQR
          │  Stage 2: EP filter                 │  wavefront planarity
          │  Stage 3: Multi-hop removal         │  2F / 3F ground reflections
          │  Stage 4: DBSCAN noise rejection    │  (f, h, V*, A, EP) cluster
          │  Stage 5: RANSAC trace fitting      │  polynomial h*(f) outlier removal
          │  Stage 6: Temporal coherence        │  multi-scan cell occupancy
          └────────────────────────────────────┘
                   │
                   ▼
             pd.DataFrame  (filtered echoes, ``sounding_index`` column added)
```

## Classes

::: pynasonde.vipir.riq.parsers.filter.IonogramFilter
    handler: python
    options:
        show_root_heading: true
        show_source: false
        members:
            - __init__
            - filter
            - summary
            - stats

---

## Constructor parameters

### Stage 1 — RFI blanking

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rfi_enabled` | `bool` | `True` | Enable RFI frequency blanking |
| `rfi_height_iqr_km` | `float` | `300.0` | Flag a frequency if its echo height IQR exceeds this value (km) |
| `rfi_min_echoes` | `int` | `3` | Minimum echoes at a frequency before the height-spread test applies |

Detection is based on **height spread**, not echo count.  Count-based
detection fails when `EchoExtractor` caps the number of echoes per pulset
(e.g. `max_echoes_per_pulset=5`), because both ionospheric and RFI
frequencies then return the same number of echoes.

RFI illuminates random gates across all heights → height IQR ≈ 300–800 km.
Ionospheric echoes cluster near E/F-layer heights → height IQR < 150 km.

### Stage 2 — EP (wavefront residual) filter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ep_filter_enabled` | `bool` | `True` | Enable EP filter |
| `ep_max_deg` | `float` | `90.0` | Maximum allowed planar-wavefront residual (degrees) |

The EP parameter is the RMS residual of the least-squares fit of inter-antenna
phase differences to a planar wavefront model.  A large EP indicates a
non-planar (multipath, distorted, or RFI-contaminated) wavefront.  Use a
conservative threshold (90°) — oblique real echoes routinely reach 50–80°
at low SNR; let Stage 4 DBSCAN handle subtler cases.

### Stage 3 — Multi-hop removal

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `multihop_enabled` | `bool` | `True` | Enable multi-hop (2F, 3F) removal |
| `multihop_orders` | `tuple[int, ...]` | `(2, 3)` | Hop orders to test |
| `multihop_height_tol_km` | `float` | `50.0` | Height tolerance for Nh* matching (km) |
| `multihop_snr_margin_db` | `float` | `6.0` | Minimum amplitude deficit of multi-hop vs 1F echo (dB) |

At each frequency, the 1F reference is the **strongest** echo in the lower half
of the height distribution (echoes at or below the median height).  This is more robust than
taking the minimum-height echo, which could be a stray noise point.  Echoes near N × h*(1F)
(within `multihop_height_tol_km`) that are also at least `multihop_snr_margin_db` weaker than
the 1F echo are labelled as N-hop artefacts and removed.

### Stage 4 — DBSCAN clustering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dbscan_enabled` | `bool` | `True` | Enable DBSCAN noise rejection |
| `dbscan_eps` | `float` | `1.0` | DBSCAN neighbourhood radius in normalised feature space |
| `dbscan_min_samples` | `int` | `5` | Minimum cluster size |
| `dbscan_features` | `tuple[str, ...]` | see below | DataFrame columns to use as DBSCAN features |

Default features:
```python
("frequency_khz", "height_km", "velocity_mps", "amplitude_db", "residual_deg")
```

Each feature is normalised by its inter-quartile range before DBSCAN so that
all dimensions have comparable weight.  Echoes assigned cluster label `−1`
(noise) are rejected.

### Stage 5 — RANSAC trace fitting

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ransac_enabled` | `bool` | `True` | Enable RANSAC polynomial trace fitting |
| `ransac_residual_km` | `float` | `100.0` | Maximum height residual for an echo to be an inlier (km) |
| `ransac_min_samples` | `int` | `10` | Echoes randomly sampled per RANSAC iteration |
| `ransac_n_iter` | `int` | `200` | Number of RANSAC iterations per sounding |
| `ransac_poly_degree` | `int` | `3` | Polynomial degree for the h*(f) trace model |
| `ransac_min_inlier_fraction` | `float` | `0.3` | Minimum inlier fraction for a model to be accepted |

Fits a degree-`ransac_poly_degree` polynomial h*(f) to the (frequency, height) echo cloud using
Random Sample Consensus.  Echoes further than `ransac_residual_km` from the best-fit curve are
rejected as outliers.  Run independently per sounding index so that each sounding's ionospheric
trace is fitted separately.  If no iteration achieves `ransac_min_inlier_fraction` of the active
echoes the stage is skipped for that sounding.

### Stage 6 — Temporal coherence (multi-sounding only)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temporal_enabled` | `bool` | `True` | Enable temporal coherence filter |
| `temporal_min_soundings` | `int` | `3` | Minimum soundings a cell must appear in |
| `temporal_freq_bin_khz` | `float` | `50.0` | Frequency bin width for cell definition (kHz) |
| `temporal_height_bin_km` | `float` | `50.0` | Height bin width for cell definition (km) |

This stage is silently skipped when only one sounding is supplied.

---

## Quick start

### Single sounding

```python
from pynasonde.vipir.riq.echo import EchoExtractor
from pynasonde.vipir.riq.parsers.filter import IonogramFilter
from pynasonde.vipir.riq.parsers.read_riq import VIPIR_VERSION_MAP, RiqDataset

riq = RiqDataset.create_from_file(
    "WI937_2022233235902.RIQ",
    unicode="latin-1",
    vipir_config=VIPIR_VERSION_MAP.configs[1],
)
ext = EchoExtractor(
    sct=riq.sct, pulsets=riq.pulsets,
    snr_threshold_db=3.0, min_height_km=60.0, max_height_km=1000.0,
).extract()

filt = IonogramFilter(
    ep_max_deg=45.0,
    dbscan_eps=1.0,
    dbscan_min_samples=5,
    temporal_enabled=False,      # only one sounding
)

df_clean = filt.filter(ext)     # accepts single extractor or list
print(filt.summary())
```

### Multiple soundings (temporal coherence)

```python
extractors = []
for fname in riq_file_list:
    riq = RiqDataset.create_from_file(fname, ...)
    ext = EchoExtractor(...).extract()
    extractors.append(ext)

filt = IonogramFilter(
    temporal_enabled=True,
    temporal_min_soundings=3,
    temporal_freq_bin_khz=50.0,
    temporal_height_bin_km=50.0,
)

df_clean = filt.filter(extractors)          # list of extractors
# df_clean has column "sounding_index" = 0, 1, 2, ...
```

---

## Output DataFrame columns

The returned DataFrame contains all columns from
:meth:`~pynasonde.vipir.riq.echo.EchoExtractor.to_dataframe` plus:

| Column | Type | Description |
|--------|------|-------------|
| `sounding_index` | `int` | Index into the input extractor list (0 for single sounding) |

---

## Statistics

After calling :meth:`filter`, the `stats` attribute is populated:

```python
{
    "rfi":       {"input": N, "rejected": N_rfi},
    "ep":        {"input": N, "rejected": N_ep},
    "multihop":  {"input": N, "rejected": N_mh},
    "dbscan":    {"input": N, "rejected": N_db},
    "temporal":  {"input": N, "rejected": N_t},   # absent for single sounding
    "summary":   {"total_input": N, "total_kept": N_k},
}
```

Human-readable via :meth:`summary`:

```python
print(filt.summary())
# Stage         Input  Rejected  Kept   Retention
# ─────────────────────────────────────────────────
# RFI            3206        12  3194     99.6 %
# EP             3194       281  2913     91.2 %
# Multi-hop      2913        87  2826     97.0 %
# DBSCAN         2826       179  2647     93.7 %
# ─────────────────────────────────────────────────
# Total          3206       559  2647     82.6 %
```

---

## Algorithm notes

### RFI height-spread test

```
h_iqr(f) = IQR of height_km at frequency f
flag f  if  h_iqr(f) > rfi_height_iqr_km  AND  count(f) >= rfi_min_echoes
```

Detection is based on the **height spread** of echoes at each frequency, not echo count.
Count-based detection is unreliable when `max_echoes_per_pulset` caps the per-frequency
echo count.  RFI illuminates random range gates → height IQR ≈ 300–800 km; ionospheric
echoes cluster near E/F-layer heights → height IQR < 150 km.

### Multi-hop geometry

A 2F echo appears at exactly twice the virtual height of the 1F echo at
the same frequency because it undergoes an extra ground bounce:

```
h*(2F) ≈ 2 × h*(1F)
A(2F)  ≈ A(1F) − 10 to 20 dB
```

The filter identifies the strongest echo in the **lower half** of the height
distribution at each frequency as the 1F reference, then flags any echo at
N × h*(1F) ± `multihop_height_tol_km` that is also weaker by at least
`multihop_snr_margin_db`.

### DBSCAN feature scaling

Each feature column is independently normalised:

```python
x_norm = (x - median(x)) / IQR(x)
```

This makes the `dbscan_eps` parameter approximately equivalent to
"number of IQR units" of separation, giving it a physical interpretation
independent of the units of each parameter.

### Temporal coherence cell occupancy

```
cell(i) = (freq_bin, height_bin)  for echo i
occupancy(cell) = number of soundings containing ≥ 1 echo in cell
keep echo i  iff  occupancy(cell(i)) >= temporal_min_soundings
```

---

## References

- Zabotin N. A. et al. (2006). NeXtYZ: Three-dimensional electron density
  inversion for dynasonde ionograms. *Radio Science* 41(6).
  <https://doi.org/10.1029/2005RS003352>

- Ester M. et al. (1996). A density-based algorithm for discovering clusters
  in large spatial databases with noise. *KDD-96 Proceedings*.

---

## Related

- [Echo Extractor API](../echo.md)
- [Filter examples](../../../../examples/vipir/ionogram_filter.md)
- `examples/vipir/ionogram_filter_wi937.py`
- `examples/vipir/ionogram_filter_pl407.py`
- `examples/vipir/ionogram_filter_multi.py`
