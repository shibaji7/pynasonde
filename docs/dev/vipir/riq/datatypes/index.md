# RIQ Datatypes (pynasonde.vipir.riq.datatypes)

This section documents the dataclasses that represent RIQ configuration tables.

Contents

- `sct.md` — `SctType`: system configuration table with general, station, timing, frequency, receiver, exciter, and monitor sub-structs
- `pct.md` — `PctType`: pulse configuration table

Example quick start

```py
from pynasonde.vipir.riq.datatypes.sct import SctType

sct = SctType()
print(sct.station.rx_count)
sct.dump_sct()
```
