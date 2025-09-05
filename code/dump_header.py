fname = (
    "/home/chakras4/Research/ERAUCodeBase/readriq-2.08/Justin/PL407_2024058061501.RIQ"
)
import numpy as np

# fname = "tmp/WI937_2022233235902.RIQ"
from pynasonde.vipir.ngi.plotlib import Ionogram
from pynasonde.vipir.riq.datatypes.pct import PctType
from pynasonde.vipir.riq.datatypes.sct import SctType
from pynasonde.vipir.riq.parsers.read_riq import (
    VIPIR_VERSION_MAP,
    RiqDataset,
    adaptive_gain_filter,
    remove_morphological_noise,
)

# x, y = SctType(), PctType()
# x.read_sct(fname)
# x.dump_sct("tmp/WI937_2022233235902_sct.txt")
# y.load_sct(x)
# y.read_pct(fname)
# y.dump_pct(to_file="tmp/PL407_2024058061501_pct.txt")

# pri = PriType()
# pri.load_data(fname)
if True:
    riq = RiqDataset.create_from_file(
        fname, unicode="latin-1", vipir_config=VIPIR_VERSION_MAP.configs[0]
    )
    ion = adaptive_gain_filter(
        riq.get_ionogram(threshold=50, remove_baseline_noise=True),
        apply_median_filter=True,
        median_filter_size=3,
    )
    # # print(i.frequencies.tolist())
    p = Ionogram(ncols=1, nrows=1)
    # a = i.amplitude
    # # a[a <= 0] = np.nan
    # # a = np.ma.masked_invalid(a)
    p.add_ionogram(
        frequency=ion.frequency,
        height=ion.height,
        value=ion.powerdB,
        mode="O",
        xlabel="Frequency, MHz",
        ylabel="Virtual Height, km",
        ylim=[50, 1000],
        xlim=[1.8, 22],
        add_cbar=True,
        cbar_label="Power, dB",
        prange=[0, 100],
        del_ticks=False,
    )
    p.save("tmp/PL407_2024058061501.png")
    p.close()
# from pynasonde.pynasonde.ngi.source import DataSource
# print(VIPIR_VERSION_MAP.configs[1])
# fname = "tmp/WI937_2022233235902.RIQ"
# riq = RiqDataset.create_from_file(
#     fname, unicode="latin-1", vipir_config=VIPIR_VERSION_MAP.configs[1]
# )
# ion = riq.get_ionogram(prominence=200, bins=30)
# # # # # print(i.frequencies.tolist())
# # (snr, frequencies, heights) = riq.get_ionogram()
# p = Ionogram(ncols=1, nrows=1)
# # # a = i.amplitude
# # # # a[a <= 0] = np.nan
# # # # a = np.ma.masked_invalid(a)
# p.add_ionogram(
#     frequency=ion.frequency,
#     height=ion.height,
#     value=ion.powerdB,
#     mode="O",
#     xlabel="Frequency, MHz",
#     ylabel="Virtual Height, km",
#     ylim=[0, 1000],
#     xlim=[1.8, 22],
#     add_cbar=True,
#     cbar_label="O-mode Power, dB",
#     prange=[0, 40],
#     del_ticks=False,
# )
# p.save("tmp/WI937_2022233235902.png")
# p.close()

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

plt.figure(figsize=(8, 4))
import pandas as pd

s = pd.Series(ion.powerdB.flatten())
clean_s = s.replace([np.inf, -np.inf], np.nan).dropna()
counts, bin_edges = np.histogram(clean_s, bins=100)
# Find local minima (dips) by finding peaks in the negative counts
dips, _ = find_peaks(-counts, prominence=100)  # Adjust prominence for noise rejection

dip_bins = bin_edges[dips]
print("Detected dips at:", dip_bins)

# If you want the dip closest to 15:
closest_dip = dip_bins[np.argmin(np.abs(dip_bins - 15))]
print("Closest dip to 15:", closest_dip)

# Optional: plot
plt.hist(clean_s, bins=100, alpha=0.5, label="Power dB")
for dip in dip_bins:
    plt.axvline(dip, color="r", linestyle="--")
# plt.hist(ion.power.flatten(), bins=1000, alpha=0.5, label="Power")
# plt.xscale("log")


plt.yscale("log")
plt.xlabel("Power")
plt.ylabel("Count")
plt.legend()
plt.title("Ionogram Power Distribution")
plt.tight_layout()
plt.savefig("tmp/power_distribution.png")
plt.close()
# ds = DataSource(source_folder="./tmp/20240408/")
# # # print(ds.file_paths)
# ds.load_data_sets()
# print(ds.datasets[0].X_mode_power)
# for f in range(2, 9):
#     print(ds.extract_FTI_RTI(flim=[f - 0.1, f + 0.1], index=f))
# ds.extract_ionograms()


# from pynasonde.webhook import Webhook
# print(Webhook().download(itr=-1))
