# DVL parser

Parses DVL-format digisonde outputs.

::: pynasonde.digisonde.parsers.dvl
    handler: python
    options:
        show_root_heading: true
        show_source: false

::: pynasonde.digisonde.parsers.dvl.DvlExtractor
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - __init__
            - extract
            - read_file
            - extract_DVL_pandas
            - load_DVL_files
            
