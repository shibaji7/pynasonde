# MMM parser

Parses MMM (multi-mode) digisonde outputs.

::: pynasonde.digisonde.parsers.mmm
    handler: python
    options:
        show_root_heading: true
        show_source: false

## Example

```python
from pynasonde.digisonde.parsers.mmm import MmmExtractor
m = MmmExtractor()
# m.parse(path)
# df = m.to_pandas()
```
