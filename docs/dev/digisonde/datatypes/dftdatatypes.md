## DFT datatypes

Dataclasses modeling DFT-format headers and spectral blocks.

::: pynasonde.digisonde.datatypes.dftdatatypes
    handler: python
    options:
        show_root_heading: true
        show_source: false

## Example

```python
import numpy as np
from pynasonde.digisonde.datatypes.dftdatatypes import DftHeader, DopplerSpectra

hdr = DftHeader(year=2024, num_doppler_lines=64)
spec = DopplerSpectra(amplitude=np.zeros(64), phase=np.zeros(64))
```
