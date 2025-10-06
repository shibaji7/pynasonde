# Sky parser

Parses sky images and associated metadata.

::: pynasonde.digisonde.parsers.sky
    handler: python
    options:
        show_root_heading: true
        show_source: false


::: pynasonde.digisonde.parsers.sky.SkyExtractor
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - __init__
            - read_file
            - parse_line
            - parse_data_header
            - parse_freq_header
            - extract
            - parse_sky_data
            - get_doppler_freq
            - to_pandas
