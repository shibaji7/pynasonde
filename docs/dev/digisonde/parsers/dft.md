# DFT parser

Parses DFT-format digisonde outputs.

::: pynasonde.digisonde.parsers.dft
    handler: python
    options:
        show_root_heading: true
        show_source: false

::: pynasonde.digisonde.parsers.dft.DftExtractor
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - __init__
            - extract
            - extract_header_from_amplitudes
            - to_int
            - unpack_7_1
