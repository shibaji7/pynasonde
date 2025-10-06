# RSF parser

Parses RSF digisonde outputs.

::: pynasonde.digisonde.parsers.rsf
    handler: python
    options:
        show_root_heading: true
        show_source: false

::: pynasonde.digisonde.parsers.rsf.RsfExtractor
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - __init__
            - extract
            - add_dicts_selected_keys
            - unpack_5_3
            - unpack_bcd