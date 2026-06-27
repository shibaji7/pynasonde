"""Debug CADI receiver geometry, O/X mode, and AOA for one echo.

Use this script when ``cadi_interferometry_plot.py`` produces unexpected O/X
labels or ``XL``/``YL`` projections. It prints the file-order receiver mapping,
physical CADI geometry, baseline phase differences, O/X classification, and AOA
solution for one selected detection row.
"""

from pprint import pprint

from pynasonde.digisonde.cadi import (
    CadiArray,
    CadiExtractor,
    CadiReceiverLayout,
    debug_echo_geometry,
)


file = "tmp/CADI/6E131200.md4"
row_index = 1652

# Station/location.ini metadata.
lat = 17.47
lon = 78.57
diagonal_m = 30.0
orientation_deg = 0.0
rx_bitmask = 14  # File rx1/rx2/rx3 map to physical Rx2/Rx3/Rx4.

# Standard NH CADI layout: Rx1 north, Rx2 east, Rx3 south, Rx4 west.
receiver_layout = CadiReceiverLayout.STANDARD


df = CadiExtractor.extract_CADI(file, product="products")
if df.empty:
    raise SystemExit(f"No CADI detections found in {file}.")
if row_index >= len(df):
    raise SystemExit(f"row_index={row_index} is outside dataframe length {len(df)}.")

array = CadiArray.from_receiver_count(
    n_receivers=3,
    lat=lat,
    lon=lon,
    diagonal_m=diagonal_m,
    orientation_deg=orientation_deg,
    rx_bitmask=rx_bitmask,
    receiver_layout=receiver_layout,
)

row = df.iloc[row_index]
debug = debug_echo_geometry(row, array)

print("\nSelected row")
print("------------")
print(row[["frequency_mhz", "height_km", "doppler_bin", "mean_power_db"]])

print("\nReceiver mapping")
print("----------------")
pprint(debug["receiver_mapping"])

print("\nReceiver geometry")
print("-----------------")
pprint(debug["array"]["receivers"])

print("\nBaselines")
print("---------")
pprint(debug["array"]["baselines"])

print("\nBaseline phases")
print("---------------")
pprint(debug["baseline_phases"])

print("\nO/X result")
print("----------")
print({"mode": debug["mode"], "polarization_deg": debug["polarization_deg"]})

print("\nAOA result")
print("----------")
pprint(debug["aoa"])
