# Digisonde Autoscaler (`pynasonde.digisonde.scaler`)

DataFrame-native ionogram scaling for Digisonde data. RSF, SBF and MMM parser
output is first normalised to a canonical echo schema, then filtered
(vertical RFI), traced (Viterbi) and scaled (foE/foF2, hmF2 and friends).

!!! note "Install"
    `schema` and `solar` need only the core install. Filtering, tracing and
    scaling need `pip install "pynasonde[digisonde-scaler]"`.

| Module | Summary |
|--------|---------|
| [schema](schema.md) | Canonical echo schema: detect and normalise RSF/SBF/MMM DataFrames |
| [rfi_filter](rfi_filter.md) | Vertical RFI filters (adaptive amplitude gate, neighbour-predicted, column density) |
| [core](core.md) | `IonogramAutoScaler`, `IonoscaleResult` and the `scale_dataframe*` entry points |
| [viterbi](viterbi.md) | `ViterbiTracer` trace extraction |
| [hmF2](hmF2.md) | `ScaleHmF2` peak height scaling |
| [presets](presets.md) | Tuning presets and solar parameter sets (`get_viterbi_preset`) |
| [solar](solar.md) | Solar zenith angle and day/twilight/night regime helpers |
| [postprocess](postprocess.md) | `despike_scaled_parameters` post-processing |

## Quick start

```python
from pynasonde.digisonde.scaler import scale_dataframe_with_preset

# df: echo DataFrame for one ionogram (RSF / SBF / MMM or canonical schema)
result = scale_dataframe_with_preset(
    df,
    time=sounding_time,   # used for the solar zenith angle / day-twilight-night regime
    lat_deg=37.8,
    lon_deg=-75.4,
    tuning_preset="balanced",
)  # -> IonoscaleResult  (pass return_details=True to also get per-layer traces)
```

See the [RSF Viterbi autoscaling example](../../../examples/digisonde/rsf_scaler.md).
