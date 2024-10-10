from pynasonde.pynasonde.ngi.source import DataSource

ds = DataSource(source_folder="./tmp/20240407/")
ds.load_data_sets(800, 801)
from pynasonde.ngi.scale import AutoScaler

for i, dx in enumerate(ds.datasets):
    a = AutoScaler(dx)
    a.image_segmentation()
    a.to_binary_traces()
    a.draw_sanity_check_images("tmp/scan.png")
# ds.extract_ionograms()
# ds.save_scaled_dataset()
