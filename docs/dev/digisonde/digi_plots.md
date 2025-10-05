::: pynasonde.digisonde.digi_plots.DigiPlots
    handler: python
    options:
        show_root_heading: true
        show_source: false
        members:
            - __init__
            - get_axes
            - save
            - close
            - _add_colorbar

::: pynasonde.digisonde.digi_plots.SaoSummaryPlots
    handler: python
    options:
        show_root_heading: true
        show_source: false
        members:
            - add_TS
            - plot_TS
            - plot_ionogram
            - plot_isodensity_contours


## Examples

```py
from pynasonde.digisonde.digi_plots import SaoSummaryPlots

plotter = SaoSummaryPlots(fig_title='Example', nrows=1, ncols=1)
# Use plotter.add_TS(df) where df is a DataFrame produced by parser(s)
```