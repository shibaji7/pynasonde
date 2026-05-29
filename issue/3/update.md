Update for issue #3 (`Prob in Reading/Accessing .SAO files`)

Implemented and pushed in `v1.2.2`:
- SAO parser now supports both single-record and multi-entry/day-file `.SAO`.
- Added per-record UTC datetime parsing from `FF...` lines.
- Added SAO parse modes:
  - `mode="auto"` (detect single vs multi)
  - `mode="single"` (supports `record_index`)
  - `mode="multi"` (parse all records)
- Propagated mode/index handling through:
  - `extract_SAO(...)`
  - `load_SAO_files(...)`
- Added/updated tests for these paths.

Suggested usage:
- Use the new advanced example:
  - `examples/digisonde/sao_multi.py`
- Before running, update:
  - `folders = [...]` to your local SAO/day-file directory
  - `date = ...` to your target day
  - optional: `selected_record_index = ...` for single-scan extraction from a day file

Patch/reference artifacts:
- SAO-only patch: `issue/3/sao_issue.patch`
- Issue update notes: `issue/3/update.md`

Figures (repo links):
- [PynasondeV1-SAO-Multi-01.png](https://raw.githubusercontent.com/shibaji7/pynasonde/main/docs/examples/figures/PynasondeV1-SAO-Multi-01.png)
- [PynasondeV1-SAO-Multi-02.png](https://raw.githubusercontent.com/shibaji7/pynasonde/main/docs/examples/figures/PynasondeV1-SAO-Multi-02.png)

Preview:

![PynasondeV1-SAO-Multi-01](https://raw.githubusercontent.com/shibaji7/pynasonde/main/docs/examples/figures/PynasondeV1-SAO-Multi-01.png)

![PynasondeV1-SAO-Multi-02](https://raw.githubusercontent.com/shibaji7/pynasonde/main/docs/examples/figures/PynasondeV1-SAO-Multi-02.png)
