# SAO parser

Parses SAO XML Digisonde files.

::: pynasonde.digisonde.parsers.sao
    handler: python
    options:
        show_root_heading: true
        show_source: false

## Example

```python
from pynasonde.digisonde.parsers.sao import SaoExtractor
x = SaoExtractor()
# x.parse(xml_string)
# df = x.to_pandas()
```
