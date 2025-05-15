fname = (
    "/home/chakras4/Research/ERAUCodeBase/readriq-2.08/Justin/PL407_2024058061501.RIQ"
)
from pynasonde.riq.headers.pct import PctType
from pynasonde.riq.headers.pri import PriType
from pynasonde.riq.headers.sct import SctType
from pynasonde.riq.load import RiqDataset

x, y = SctType(), PctType()
x.read_sct(fname)
x.dump_sct("tmp/PL407_2024058061501_sct.txt")
y.load_sct(x)
y.read_pct(fname)
y.dump_pct(to_file="tmp/PL407_2024058061501_pct.txt")

pri = PriType()
pri.load_data(fname)

# from pynasonde.pynasonde.ngi.source import DataSource

# ds = DataSource(source_folder="./tmp/20240408/")
# # # print(ds.file_paths)
# ds.load_data_sets()
# print(ds.datasets[0].X_mode_power)
# for f in range(2, 9):
#     print(ds.extract_FTI_RTI(flim=[f - 0.1, f + 0.1], index=f))
# ds.extract_ionograms()


# from pynasonde.webhook import Webhook
# print(Webhook().download(itr=-1))
