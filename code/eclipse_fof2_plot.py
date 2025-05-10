import datetime as dt

from pynasonde.ngi.source import DataSource, Trace

date = dt.datetime(2024, 4, 9)

ds = DataSource(source_folder=f"./tmp/{date.strftime('%Y%m%d')}/")
ds.load_data_sets(0, 1)
from pynasonde.ngi.utils import load_toml

cfg = load_toml()

from pynasonde.ngi.scale import AutoScaler, NoiseProfile

for i, dx in enumerate(ds.datasets):
    scaler = AutoScaler(
        dx,
        noise_profile=NoiseProfile(constant=cfg.ngi.scaler.noise_constant),
        mode=cfg.ngi.scaler.mode,
        filter=dict(
            frequency=[cfg.ngi.scaler.frequency_min, cfg.ngi.scaler.frequency_max],
            height=[cfg.ngi.scaler.height_min, cfg.ngi.scaler.height_max],
        ),
        apply_filter=cfg.ngi.scaler.apply_filter,
        segmentation_method=cfg.ngi.scaler.segmentation_method,
    )
    scaler.mdeian_filter()
    scaler.image_segmentation()
    scaler.to_binary_traces(
        nbins=cfg.ngi.scaler.otsu.nbins,
        thresh=cfg.ngi.scaler.otsu.thresh,
        eps=cfg.ngi.scaler.dbscan.eps,
        min_samples=cfg.ngi.scaler.dbscan.min_samples,
    )
    scaler.draw_sanity_check_images(f"tmp/scan_{i}.png", font_size=15)
    del scaler
    break
ds.save_scaled_parameters(
    {
        "mode": cfg.ngi.scaler.mode,
        "noise_profile": f"Constant/{cfg.ngi.scaler.noise_constant}",
        "Otsu.nbins": cfg.ngi.scaler.otsu.nbins,
        "Otsu.thresh": cfg.ngi.scaler.otsu.thresh,
        "DBSCAN.eps": cfg.ngi.scaler.dbscan.eps,
        "DBSCAN.min_samples": cfg.ngi.scaler.dbscan.min_samples,
        "segmentation_method": cfg.ngi.scaler.segmentation_method,
        "frequency_lims": [cfg.ngi.scaler.frequency_min, cfg.ngi.scaler.frequency_max],
        "height_lims": [cfg.ngi.scaler.height_min, cfg.ngi.scaler.height_max],
    },
    mode=cfg.ngi.scaler.mode,
)
Trace.load_saved_scaled_parameters(f"./tmp/{date.strftime('%Y%m%d')}/scaled/")
