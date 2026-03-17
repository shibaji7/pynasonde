# RIQ Parsers (pynasonde.vipir.riq.parsers)

This section documents the RIQ parser modules.

Contents

- `read_riq.md` — low-level RIQ binary reader, threshold detection, morphological noise removal, adaptive gain filter, `Pulset` container
- `trace.md` — IQ-data phase computation, echo-trace extraction, phase-velocity estimation

Example quick start

```py
from pynasonde.vipir.riq.parsers.trace import compute_phase
import numpy as np

i = np.array([1.0, 0.0, -1.0])
q = np.array([0.0, 1.0,  0.0])
phase = compute_phase(i, q)
```
