# DFT parser

Parses DFT-format digisonde outputs.

::: pynasonde.digisonde.parsers.dft
    handler: python
    options:
        show_root_heading: true
        show_source: false

## Example

```python
from pynasonde.digisonde.parsers.dft import DftExtractor
d = DftExtractor()
# d.parse(path)
# df = d.to_pandas()
```
