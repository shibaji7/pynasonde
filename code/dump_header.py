fname = (
    "/home/chakras4/Research/ERAUCodeBase/readriq-2.08/Justin/PL407_2024058061501.RIQ"
)
import numpy as np

# fname = "tmp/WI937_2022233235902.RIQ"
from pynasonde.vipir.ngi.plotlib import Ionogram
from pynasonde.vipir.riq.headers.pct import PctType
from pynasonde.vipir.riq.headers.pri import PriType
from pynasonde.vipir.riq.headers.sct import SctType
from pynasonde.vipir.riq.load import VIPIR_VERSION_MAP, RiqDataset

# x, y = SctType(), PctType()
# x.read_sct(fname)
# x.dump_sct("tmp/WI937_2022233235902_sct.txt")
# y.load_sct(x)
# y.read_pct(fname)
# y.dump_pct(to_file="tmp/PL407_2024058061501_pct.txt")

# pri = PriType()
# pri.load_data(fname)
riq = RiqDataset.create_from_file(
    fname, unicode="latin-1", vipir_version=VIPIR_VERSION_MAP.One
)
i = riq.ionogram()
# print(i.frequencies.tolist())
p = Ionogram(ncols=1, nrows=1)
a = i.amplitude
# a[a <= 0] = np.nan
# a = np.ma.masked_invalid(a)
p.add_ionogram(
    frequency=i.frequencies,
    height=i.range_gates,
    value=a,
    mode="O",
    xlabel="Frequency, MHz",
    ylabel="Virtual Height, km",
    ylim=[50, 600],
    xlim=[1.8, 22],
    add_cbar=True,
    cbar_label="O-mode Power, dB",
    prange=[0, 100],
    del_ticks=False,
)
p.save("tmp/PL407_2024058061501.png")
p.close()
# from pynasonde.pynasonde.ngi.source import DataSource

fname = "tmp/WI937_2022233235902.RIQ"
riq = RiqDataset.create_from_file(
    fname, unicode="latin-1", vipir_version=VIPIR_VERSION_MAP.Two
)
i = riq.ionogram()
# # print(i.frequencies.tolist())
p = Ionogram(ncols=1, nrows=1)
a = i.amplitude
# a[a <= 0] = np.nan
# a = np.ma.masked_invalid(a)
p.add_ionogram(
    frequency=i.frequencies,
    height=i.range_gates,
    value=a,
    mode="O",
    xlabel="Frequency, MHz",
    ylabel="Virtual Height, km",
    ylim=[0, 1000],
    xlim=[1, 22],
    add_cbar=True,
    cbar_label="O-mode Power, dB",
    prange=[5, 30],
    del_ticks=False,
)
p.save("tmp/WI937_2022233235902.png")
p.close()

# ds = DataSource(source_folder="./tmp/20240408/")
# # # print(ds.file_paths)
# ds.load_data_sets()
# print(ds.datasets[0].X_mode_power)
# for f in range(2, 9):
#     print(ds.extract_FTI_RTI(flim=[f - 0.1, f + 0.1], index=f))
# ds.extract_ionograms()


# from pynasonde.webhook import Webhook
# print(Webhook().download(itr=-1))
