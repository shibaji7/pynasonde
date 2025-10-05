# EDP parser

Parses EDP format files.

::: pynasonde.digisonde.parsers.edp
    handler: python
    options:
        show_root_heading: true
        show_source: false

## Example

```python
from pynasonde.digisonde.parsers.edp import EdpExtractor
e = EdpExtractor()
# e.parse(path)
# df = e.to_pandas()
```
