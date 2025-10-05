# Sky parser

Parses sky images and associated metadata.

::: pynasonde.digisonde.parsers.sky
    handler: python
    options:
        show_root_heading: true
        show_source: false

## Example

```python
from pynasonde.digisonde.parsers.sky import SkyExtractor
s = SkyExtractor()
# s.parse(file_path)
# df = s.to_pandas()
```
