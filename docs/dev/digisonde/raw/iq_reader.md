# IQ File Reader — `iq_reader`

<span class="api-badge api-package">P</span>
`pynasonde.digisonde.raw.iq_reader` — streaming reader for time-partitioned one-second IQ binary files.

---

## IQStream

<span class="api-badge api-class">C</span>
Mirrors the Julia `IQStream` object.  Manages file handles across second boundaries and exposes a simple `read_samples()` interface.

::: pynasonde.digisonde.raw.iq_reader.IQStream
    handler: python
    options:
        show_root_heading: true
        show_source: false
        members:
            - read_samples
            - close

---

## Module-level helpers

<span class="api-badge api-method">M</span>
`get_frequencies` — scan the directory tree and return center / sample frequencies from the first `.bin` filename found.

::: pynasonde.digisonde.raw.iq_reader.get_frequencies
    handler: python
    options:
        show_root_heading: true
        show_source: false

<span class="api-badge api-method">M</span>
`get_channels` — list the unique channel tags (e.g. `ch0`, `ch1`) present in the current minute directory.

::: pynasonde.digisonde.raw.iq_reader.get_channels
    handler: python
    options:
        show_root_heading: true
        show_source: false

---

## File layout

IQ recordings follow the naming convention:

```
<root>/YYYY-mm-dd/HH/MM/<timestamp>_<channel>_fc<kHz>kHz_bw<kHz>kHz.bin
```

Each `.bin` file contains exactly one second of interleaved 16-bit little-endian I and Q samples (4 bytes per complex sample).

## Quick start

```python
import datetime as dt
from pynasonde.digisonde.raw.iq_reader import IQStream, get_channels

epoch = dt.datetime(2023, 10, 14, 16, 0, 0, tzinfo=dt.timezone.utc)
dir_iq = "/media/chakras4/69F9D939661D263B"

# Discover available channels
channels = get_channels(dir_iq, epoch)
print(channels)  # e.g. ['ch0', 'ch1']

# Open a stream and read one second of samples
stream = IQStream(dir_iq, epoch, rx_tag="ch0")
samples = stream.read_samples(epoch)   # np.ndarray of np.complex64
stream.close()
```
