# RSF datatypes

::: pynasonde.digisonde.datatypes.rsfdatatypes
    handler: python
    options:
        show_root_heading: true
        show_source: false

## Example

```python
## RSF datatypes

Dataclasses for RSF-format records and frequency groups.

::: pynasonde.digisonde.datatypes.rsfdatatypes
    handler: python
    options:
        show_root_heading: true
        show_source: false

## Example

```python
from pynasonde.digisonde.datatypes.rsfdatatypes import RsfHeader, RsfDataUnit
hdr = RsfHeader(year=2024, start_frequency=10.0)
unit = RsfDataUnit(header=hdr, frequency_groups=[])
```
```
