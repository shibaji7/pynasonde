# DVL parser

Parses DVL-format digisonde outputs.

::: pynasonde.digisonde.parsers.dvl
    handler: python
    options:
        show_root_heading: true
        show_source: false

## Example

```python
from pynasonde.digisonde.parsers.dvl import DvlExtractor
v = DvlExtractor()
# v.parse(path)
# df = v.to_pandas()
```
