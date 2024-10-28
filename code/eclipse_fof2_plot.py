from pynasonde.ngi.source import DataSource, Trace

ds = DataSource(source_folder="./tmp/20240407/")
ds.load_data_sets(0, -1)
from pynasonde.ngi.utils import load_toml

cfg = load_toml()

from pynasonde.ngi.scale import AutoScaler, NoiseProfile

for i, dx in enumerate(ds.datasets):
    scaler = AutoScaler(
        dx,
        noise_profile=NoiseProfile(constant=cfg.ngi.scaler.noise_constant),
        mode=cfg.ngi.scaler.mode,
    )
    scaler.image_segmentation()
    scaler.to_binary_traces(
        nbins=cfg.ngi.scaler.otsu.nbins,
        thresh=cfg.ngi.scaler.otsu.thresh,
        eps=cfg.ngi.scaler.dbscan.eps,
        min_samples=cfg.ngi.scaler.dbscan.min_samples,
    )
    scaler.draw_sanity_check_images(f"tmp/scan_{i}.png", font_size=15)
    del scaler
ds.save_scaled_parameters(
    {
        "noise_profile": f"Constant/{cfg.ngi.scaler.noise_constant}",
        "Otsu.thresh": cfg.ngi.scaler.otsu.thresh,
        "DBSCAN.eps": cfg.ngi.scaler.dbscan.eps,
        "DBSCAN.min_samples": cfg.ngi.scaler.dbscan.min_samples,
    },
    mode=cfg.ngi.scaler.mode,
)
Trace.load_saved_scaled_parameters("./tmp/20240408/scaled/")
