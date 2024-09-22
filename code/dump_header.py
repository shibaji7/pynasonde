# fname = "tmp/WI937_2024264001803.RIQ"
# from pynasonde.riq.headers.pct import PctType
# from pynasonde.riq.headers.sct import SctType

# x, y = SctType(), PctType()
# x.read_sct(fname)
# x.dump_sct("tmp/WI937_2024264001803_sct.txt")
# y.load_sct(x)
# y.read_pct(fname)
# y.dump_pct(to_file="tmp/WI937_2024264001803_pct.txt")

from pynasonde.ngi.ionograms import DataSource

ds = DataSource(source_folder="./tmp/20240920/")
# # print(ds.file_paths)
ds.load_data_sets()
# print(ds.datasets[0].X_mode_power)
print(ds.extract_FTI_RTI())
# ds.extract_ionograms()


# from pynasonde.webhook import Webhook
# print(Webhook().download(itr=-1))
