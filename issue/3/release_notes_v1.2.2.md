## pynasonde 1.2.2

Precision ionospheric radio sounding tools.

### Installation

```bash
pip install pynasonde==1.2.2
```

### Changelog

- Add multi-entry/day-file `.SAO` parsing support.
- Add per-record UTC datetime parsing from `FF...` lines.
- Add parse mode controls for SAO text files:
  - `mode="auto"`
  - `mode="single"` with `record_index`
  - `mode="multi"`
- Propagate mode/index support through `extract_SAO(...)` and `load_SAO_files(...)`.
- Extend SAO parser tests and add dedicated multi-record regression coverage.
- Add advanced day-file example: `examples/digisonde/sao_multi.py`.
