# VIPIR RIQ (pynasonde.vipir.riq)

This section documents the RIQ binary-format sub-package.

Contents

- `parsers/` — low-level binary reader and IQ-data processing functions
- `datatypes/` — `SctType` and `PctType` configuration-table dataclasses
- `utils.md` — shared RIQ utility functions

Example quick start

```py
from pynasonde.vipir.riq.datatypes.sct import SctType

sct = SctType()
sct.dump_sct(to_file="sct_dump.txt")
```
