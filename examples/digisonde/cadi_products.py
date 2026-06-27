"""Advanced CADI example: derived signal products.

This script reads CADI MD2/MD4 files and exports derived fields such as
mean power, receiver phase, and Doppler-bin index.

Update ``folders`` before running.
"""

from pathlib import Path

from pynasonde.digisonde.cadi import CadiExtractor


folders = ["tmp/CADI"]
out_csv = Path("tmp/cadi_products.csv")


df = CadiExtractor.load_CADI_files(
    folders=folders,
    product="products",
    n_procs=1,
)

if df.empty:
    print("No CADI detections found. Check folders/exts.")
else:
    keep_cols = [
        "site",
        "record_datetime",
        "frequency_mhz",
        "height_km",
        "doppler_bin",
        "mean_power_db",
        "rx1_phase_deg",
        "rx2_phase_deg",
        "rx3_phase_deg",
        "dphi_12_deg",
        "coh_12",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    print("Rows:", len(df))
    print(df[keep_cols].head(10))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
