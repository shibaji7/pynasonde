"""Basic CADI MD4/MD2 example (phase-1).

This example decodes CADI binary files and exports a per-detection table.
Update ``folders`` to your local CADI directory before running.
"""

from pathlib import Path

from pynasonde.digisonde.cadi import CadiExtractor


folders = ["tmp/CADI"]
out_csv = Path("tmp/cadi_detections.csv")


df = CadiExtractor.load_CADI_files(folders=folders, n_procs=1)

if df.empty:
    print("No CADI detections found. Check folders/exts.")
else:
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print(df[["site", "record_datetime", "frequency_mhz", "height_km"]].head(10))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
