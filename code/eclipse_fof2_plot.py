from pynasonde.ngi.source import DataSource

ds = DataSource(source_folder="./tmp/20240407/")
ds.load_data_sets(0, -1)
from pynasonde.ngi.scale import AutoScaler, NoiseProfile

for i, dx in enumerate(ds.datasets):
    scaler = AutoScaler(dx, noise_profile=NoiseProfile(constant=2))
    scaler.image_segmentation()
    scaler.to_binary_traces(thresh=3, eps=10, min_samples=100)
    scaler.draw_sanity_check_images(f"tmp/scan_{i}.png")
    del scaler
ds.save_scaled_parameters(
    {
        "noise_profile": "Constant/2",
        "thresh": 3,
        "DBSCAN.eps": 10,
        "DBSCAN.min_samples": 100,
    }
)
# ds.extract_ionograms()
# ds.save_scaled_dataset()
