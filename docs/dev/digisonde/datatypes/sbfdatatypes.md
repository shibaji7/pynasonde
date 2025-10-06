## SBF datatypes

Dataclasses for SBF-format records and frequency groups.

::: pynasonde.digisonde.datatypes.sbfdatatypes
    handler: python
    options:
        show_root_heading: true
        show_source: false

## Example

```python
from pynasonde.digisonde.datatypes.sbfdatatypes import SbfHeader, SbfDataUnit
hdr = SbfHeader(year=2024)
unit = SbfDataUnit(header=hdr, frequency_groups=[])
```
