# RSF parser

Parses RSF digisonde outputs.

::: pynasonde.digisonde.parsers.rsf
    handler: python
    options:
        show_root_heading: true
        show_source: false

## Example

```python
from pynasonde.digisonde.parsers.rsf import RsfExtractor
r = RsfExtractor()
# r.parse(path)
# df = r.to_pandas()
```
