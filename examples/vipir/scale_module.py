"""Example workflow for autoscaling VIPIR NGI ionograms.

The script mirrors the original one-off experiment but is organized into
reusable helpers and richly commented to make the processing steps easier
to follow when preparing documentation figures.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

from pynasonde.digisonde.digi_utils import setsize
from pynasonde.vipir.ngi.scale import AutoScaler, NoiseProfile
from pynasonde.vipir.ngi.source import DataSource
from pynasonde.vipir.ngi.utils import load_toml

# ---------------------------------------------------------------------------
# Constants carried over from the original script (parameters must remain unchanged).
# ---------------------------------------------------------------------------
FONT_SIZE = 15  # Matplotlib base font configured via `setsize`.
SANITY_FIGURE = "docs/examples/figures/ngi.scaler.png"
DEFAULT_STATION = "WI937"  # Station identifier retained for context.
DEFAULT_DOY = 99  # Day-of-year to process (April 8 eclipse campaign window).
DEFAULT_DATE = dt.datetime(2024, 4, 8)  # Baseline reference date (not used later).
TEMP_ROOT = Path("/tmp/vipir_fti")
DATA_ROOT = Path(
    os.environ.get(
        "VIPIR_SPEED_DEMON_ROOT",
        "/media/chakras4/Crucial X9/Solar_Eclipse_2024/public/WI937/individual/",
    )
)

# Match the example’s plotting defaults so published figures stay consistent.
setsize(FONT_SIZE)


def stage_day(root: Path, doy: int, temp_root: Path) -> Path:
    """Copy a single day of NGI ionogram files into a scratch directory."""
    # The directory naming convention in the source tree already zero-prefixes the DOY.
    src = root / "2024" / f"0{doy}" / "ionogram"
    tmp = temp_root / f"{doy}" / "ionogram"
    shutil.rmtree(tmp.parent, ignore_errors=True)
    shutil.copytree(src, tmp)
    return tmp


def load_datasets(
    folder: Path, start_idx: int, end_idx: int, jobs: int
) -> Iterable[Any]:
    """Load ionogram cubes for the desired index range."""
    datasource = DataSource(source_folder=str(folder))
    datasource.load_data_sets(start_idx, end_idx, n_jobs=jobs)
    return datasource


def build_scaler(dataset: Any, cfg: Any) -> AutoScaler:
    """Create an `AutoScaler` configured with the campaign-specific thresholds."""
    return AutoScaler(
        dataset,
        noise_profile=NoiseProfile(constant=cfg.ngi.scaler.noise_constant),
        mode=cfg.ngi.scaler.mode,
        filter={
            "frequency": [cfg.ngi.scaler.frequency_min, cfg.ngi.scaler.frequency_max],
            "height": [cfg.ngi.scaler.height_min, cfg.ngi.scaler.height_max],
        },
        apply_filter=cfg.ngi.scaler.apply_filter,
        segmentation_method=cfg.ngi.scaler.segmentation_method,
    )


def autoscale_dataset(dataset: Any, cfg: Any) -> None:
    """Run the autoscaling pipeline on a single dataset and emit the QA plot."""
    scaler = build_scaler(dataset, cfg)
    scaler.mdeian_filter()
    scaler.image_segmentation()
    scaler.to_binary_traces(
        nbins=cfg.ngi.scaler.otsu.nbins,
        thresh=cfg.ngi.scaler.otsu.thresh,
        eps=cfg.ngi.scaler.dbscan.eps,
        min_samples=cfg.ngi.scaler.dbscan.min_samples,
    )
    scaler.draw_sanity_check_images(
        SANITY_FIGURE,
        font_size=18,
        ylim=[50, 600],
        xlim=[1, 10],
        xticks=[1.5, 2.0, 3.0, 5.0, 7.0, 10.0],
        figsize=(4, 3),
        txt_color="k",
    )


def main() -> None:
    """Entry point that stages data, runs autoscaling, and cleans up."""
    config = load_toml()
    stage_folder = stage_day(DATA_ROOT, DEFAULT_DOY, TEMP_ROOT)
    try:
        datasource = load_datasets(stage_folder, 1200, 1201, jobs=20)
        for index, dataset in enumerate(datasource.datasets):
            # Processing order is preserved so repeated runs overwrite in the same sequence.
            autoscale_dataset(dataset, config)
    finally:
        # Remove the staged tree to keep working storage tidy between runs.
        shutil.rmtree(stage_folder.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
