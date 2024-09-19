from pynasonde.ngi.ionograms import DataSource

# fname = "WI937_2013169113403.RIQ"
# x, y = SctType(), PctType()
# x.read_sct(fname)
# y.load_sct(x)
# y.read_pct(fname)
# y.dump_pct()


ds = DataSource()
# print(ds.file_paths)
ds.load_data_sets()
print(ds.datasets[0].X_mode_power)
print(ds.extract_FTI_RTI())
ds.extract_ionograms()
