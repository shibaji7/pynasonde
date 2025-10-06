# EDP parser

Parses EDP format files.

::: pynasonde.digisonde.parsers.edp
    handler: python
    options:
        show_root_heading: true
        show_source: false

::: pynasonde.digisonde.parsers.edp.EdpExtractor
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - __init__
            - __update_tz__
            - read_file
            - __check_issues__
            - __parse_F2_datasets__
            - extract
            - extract_EDP
            - load_EDP_files