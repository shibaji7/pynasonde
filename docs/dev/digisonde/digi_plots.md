## DigiPlots (plotting helpers)

The `pynasonde.digisonde.digi_plots` module provides higher-level
plotting helpers used across the Digisonde parsers and tools. Below are
the main helpers with their API rendered below.

::: pynasonde.digisonde.digi_plots
    handler: python
    options:
        show_root_heading: true
        show_source: true

### Key classes

::: pynasonde.digisonde.digi_plots.DigiPlots
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - get_axes
            - save
            - close
            - _add_colorbar

::: pynasonde.digisonde.digi_plots.SaoSummaryPlots
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - add_TS
            - plot_TS
            - plot_ionogram
            - plot_isodensity_contours

::: pynasonde.digisonde.digi_plots.SkySummaryPlots
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - __init__
            - convert_to_rt
            - plot_skymap
            - plot_doppler_waterfall
            - plot_drift_velocities
            - plot_dvl_drift_velocities

::: pynasonde.digisonde.digi_plots.RsfIonogram
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - add_ionogram

## Example

```python
from pynasonde.digisonde.digi_plots import SaoSummaryPlots
plotter = SaoSummaryPlots(fig_title='Example')
# plotter.add_TS(df)
```

