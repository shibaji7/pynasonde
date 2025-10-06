## PCT datatypes

Dataclasses modeling VIPIR PCT-format data blocks.

::: pynasonde.vipir.riq.datatypes.pct
    handler: python
    options:
        show_root_heading: true
        show_source: false
        members:
            - to_mantissa_exponant

::: pynasonde.vipir.riq.datatypes.pct.Ionogram
    handler: python
    options:
        show_root_heading: true
        show_source: true

::: pynasonde.vipir.riq.datatypes.pct.PctType
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - fix_PCT_strings
            - read_pct_from_file_pointer
            - dump_pct