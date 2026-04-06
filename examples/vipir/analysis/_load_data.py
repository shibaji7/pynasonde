"""Shared data-loading helper for vipir/analysis examples.

Loads, extracts, and filters echoes from a real VIPIR RIQ file using the
standard pipeline:  RiqDataset → EchoExtractor → IonogramFilter → DataFrame.

Columns in the returned DataFrame
----------------------------------
frequency_khz, height_km, xl_km, yl_km,
polarization_deg, residual_deg, velocity_mps, amplitude_db
"""

from pynasonde.vipir.riq.echo import EchoExtractor
from pynasonde.vipir.riq.parsers.filter import IonogramFilter
from pynasonde.vipir.riq.parsers.read_riq import VIPIR_VERSION_MAP, RiqDataset

# Canonical paths relative to repository root (run scripts from repo root)
WI937 = "examples/data/WI937_2022233235902.RIQ"
PL407 = "examples/data/PL407_2024058061501.RIQ"


def load_echoes(
    fname=WI937,
    snr_threshold_db=3.0,
    ep_max_deg=90.0,
    dbscan_enabled=True,
    ransac_enabled=True,
):
    """Load, extract, and filter echoes from *fname*.

    Parameters
    ----------
    fname : str
        Path to the ``.RIQ`` file (relative to repository root).
    snr_threshold_db : float
        Minimum SNR for echo acceptance in EchoExtractor.
    ep_max_deg : float
        Maximum planar-wavefront residual (EP) to keep.
    dbscan_enabled : bool
        Whether to run the DBSCAN noise-removal stage.
    ransac_enabled : bool
        Whether to run the RANSAC trace-fit stage.

    Returns
    -------
    df : pd.DataFrame
        Filtered echo DataFrame.
    label : str
        Short human-readable label derived from the filename.
    """
    riq = RiqDataset.create_from_file(
        fname,
        unicode="latin-1",
        vipir_config=VIPIR_VERSION_MAP.configs[1],
    )
    extractor = EchoExtractor(
        sct=riq.sct,
        pulsets=riq.pulsets,
        snr_threshold_db=snr_threshold_db,
        min_height_km=60.0,
        max_height_km=1000.0,
        min_rx_for_direction=3,
        max_echoes_per_pulset=5,
    )
    extractor.extract()

    filt = IonogramFilter(
        rfi_enabled=True,
        ep_filter_enabled=True,
        ep_max_deg=ep_max_deg,
        multihop_enabled=True,
        multihop_orders=(2, 3),
        multihop_height_tol_km=50.0,
        multihop_snr_margin_db=6.0,
        dbscan_enabled=dbscan_enabled,
        dbscan_eps=1.0,
        dbscan_min_samples=5,
        dbscan_features=(
            "frequency_khz",
            "height_km",
            "velocity_mps",
            "amplitude_db",
            "residual_deg",
        ),
        ransac_enabled=ransac_enabled,
        ransac_residual_km=100.0,
        ransac_min_samples=10,
        ransac_n_iter=200,
        ransac_poly_degree=3,
        ransac_min_inlier_fraction=0.3,
        temporal_enabled=False,
    )

    df = filt.filter(extractor)

    import os

    label = os.path.basename(fname).split("_")[0]  # e.g. "WI937" or "PL407"
    return df, label
