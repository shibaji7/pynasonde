from pynasonde.ngi.source import DataSource

ds = DataSource(source_folder="./tmp/20240407/")
ds.load_data_sets(0, 10)
from pynasonde.ngi.scale import AutoScaler

for i, dx in enumerate(ds.datasets):
    a = AutoScaler(dx)
    a.image_segmentation()
    a.to_binary_traces()
    a.draw_sanity_check_images(f"tmp/scan_{i}.png")
# ds.extract_ionograms()
# ds.save_scaled_dataset()
