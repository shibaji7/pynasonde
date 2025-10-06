# DFT parser

Parses DFT-format digisonde outputs.

::: pynasonde.digisonde.parsers.dft
    handler: python
    options:
        show_root_heading: true
        show_source: true
        members:
            - extract
            - extract_header_from_amplitudes
            - to_int
            - unpack_7_1

## Example

```python
from pynasonde.digisonde.parsers.dft import DftExtractor
d = DftExtractor('filename.DFT')
# d.parse(path)
# df = d.to_pandas()
```
