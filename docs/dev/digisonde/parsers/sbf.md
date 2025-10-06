# SBF parser

Parses SBF format files produced by Digisonde systems.

::: pynasonde.digisonde.parsers.sbf
    handler: python
    options:
        show_root_heading: true
        show_source: false

## Example

```python
from pynasonde.digisonde.parsers.sbf import SbfExtractor
p = SbfExtractor()
# p.parse(path)
# df = p.to_pandas()
```
