# MMM datatypes

::: pynasonde.digisonde.datatypes.mmmdatatypes
    handler: python
    options:
        show_root_heading: true
        show_source: false

## Example

```python
## MMM / ModMax datatypes

Dataclasses used by the MMM/ModMax parser.

::: pynasonde.digisonde.datatypes.mmmdatatypes
    handler: python
    options:
        show_root_heading: true
        show_source: false

## Example

```python
from pynasonde.digisonde.datatypes.mmmdatatypes import ModMaxHeader, ModMaxDataUnit
hdr = ModMaxHeader(record_type=1, header_length=128)
block = ModMaxDataUnit(header=hdr)
```

```
