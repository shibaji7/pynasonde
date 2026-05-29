# Issue #3 Update: Prob in Reading/Accessing .SAO files

## Motivation
- Previously, SAO parsing effectively handled one record per `.SAO` file (single-scan workflow).
- In practice, many stations provide day-style `.SAO` files that contain multiple scan entries.
- As a result, only part of a day file could be parsed in earlier behavior.

## What Changed
- Added multi-entry/day-file `.SAO` support with record-by-record parsing.
- Added per-record UTC datetime parsing from each `FF...` line.
- Added parsing modes for text SAO:
  - `mode="auto"`: detect single vs multi and parse accordingly.
  - `mode="single"`: parse one selected record (supports `record_index`).
  - `mode="multi"`: parse all detected records.
- Propagated these options through wrappers:
  - `extract_SAO(...)`
  - `load_SAO_files(...)`
- Updated outputs for both extractor views:
  - `func_name="height_profile"`
  - `func_name="scaled"`
- Added advanced multi-record example:
  - `examples/digisonde/sao_multi.py`

## Test Coverage Updates
Updated tests:
- `tests/test_sao_parser.py`
- `tests/test_sao_parser_extended.py`
- `tests/test_sao_extra.py`
- `tests/test_sao_multirecord_regression.py` (new)
- `tests/test_digisonde_examples.py`

SAO-focused test subset result:
- `71 passed, 2 skipped`

## Artifacts
- SAO-only patch file: `issue/3/sao_issue.patch`
