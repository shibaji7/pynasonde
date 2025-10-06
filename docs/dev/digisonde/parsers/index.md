# Digisonde parsers

This page links individual parser modules and provides brief usage examples.

::: pynasonde.digisonde.parsers.sky.SkyExtractor
    handler: python
    options:
        show_root_heading: true
        show_source: false
        members:
            - __init__
            - extract
            - to_pandas

::: pynasonde.digisonde.parsers.sao
    handler: python
    options:
        show_root_heading: true
        show_source: false
        members:
            - parse_sao (if present)

::: pynasonde.digisonde.parsers.sbf
    handler: python
    options:
        show_root_heading: true
        show_source: false


## Examples

```py
from pynasonde.digisonde.parsers.sky import SkyExtractor

ext = SkyExtractor('path/to/file.SKY', extract_time_from_name=True)
df = ext.extract().to_pandas()
```
