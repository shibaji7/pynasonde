# Raw plots

<span class="api-badge api-class">C</span> <span class="api-badge api-method">M</span>
`pynasonde.digisonde.raw.raw_plots` — plotting helpers for raw IQ data.

Plotting helpers for raw digisonde data.

::: pynasonde.digisonde.raw.raw_plots
    handler: python
    options:
        show_root_heading: true
        show_source: false

## Example

```python
from pynasonde.digisonde.raw.raw_plots import RawPlots

rp = RawPlots(fig_title="Raw IQ", nrows=2, ncols=1, figsize=(6, 6))
ax = rp.axes[0]
# ... draw on ax (the add_pcolor and add_colorbar helpers are available) ...
rp.save("ionogram_raw.png")
rp.close()
```
