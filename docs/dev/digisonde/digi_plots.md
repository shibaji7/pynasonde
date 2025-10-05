::: pynasonde.digisonde.digi_plots
    handler: python
        options:
            show_root_heading: true
            show_source: true

::: pynasonde.digisonde.digi_plots.DigiPlots
    handler: python
    options:
        show_root_heading: true
        show_source: true
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
        show_source: true
        members:
            - add_TS
            - plot_TS
            - plot_ionogram
            - plot_isodensity_contours

