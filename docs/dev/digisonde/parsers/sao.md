# SAO parser

Parses SAO XML Digisonde files.

::: pynasonde.digisonde.parsers.sao
    handler: python
    options:
        show_root_heading: true
        show_source: false


::: pynasonde.digisonde.parsers.sao.SaoExtractor
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - __init__
            - load_XML_files
            - load_SAO_files
            - extract_SAO
            - display_struct
            - get_height_profile
            - get_scaled_datasets
            - extract
            - get_scaled_datasets_xml
            - get_height_profile_xml
            - extract_xml
            - parse_line
            - read_file