# Image parser

Parses Ionogram (.png) format files, extract data from left table.

::: pynasonde.digisonde.parsers.image_parser
    handler: python
    options:
        show_root_heading: true
        show_source: false

::: pynasonde.digisonde.parsers.image_parser.IonogramImageExtractor
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - __init__
            - extract_header
            - parse_artist_params_table
            - extract_text