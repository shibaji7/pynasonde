"""Integration-style checks that reuse code from example scripts.

These tests exercise a subset of the example workflows using the bundled sample
data under ``examples/data``. They intentionally mirror the documentation
examples while writing outputs to temporary directories to avoid polluting the
repository.
"""

from pathlib import Path

import numpy as np
import pytest

EXAMPLES_ROOT = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.skipif(
    not (EXAMPLES_ROOT / "data" / "KR835_2024099160913.SKY").exists(),
    reason="Sample SKY file not available",
)
def test_sky_example_pipeline(tmp_path):
    """Replicate the sky map example end-to-end."""
    pytest.importorskip("matplotlib")

    from pynasonde.digisonde.digi_plots import SkySummaryPlots
    from pynasonde.digisonde.parsers.sky import SkyExtractor

    data_file = EXAMPLES_ROOT / "data" / "KR835_2024099160913.SKY"
    extractor = SkyExtractor(str(data_file), True, True)
    extractor.extract()
    df = extractor.to_pandas()
    assert not df.empty

    skyplot = SkySummaryPlots()
    skyplot.plot_skymap(
        df,
        zparam="spect_dop_freq",
        text="Test\n SKY",
        cmap="Spectral",
        clim=[-0.25, 0.25],
        rlim=6,
    )
    output = tmp_path / "sky.png"
    skyplot.save(output)
    skyplot.close()

    assert output.exists() and output.stat().st_size > 0


@pytest.mark.skipif(
    not (EXAMPLES_ROOT / "data" / "PL407_2024058061501.RIQ").exists(),
    reason="Sample RIQ file not available",
)
def test_vipir_riq_example(tmp_path):
    """Exercise the VIPIR ionogram example."""
    pytest.importorskip("matplotlib")

    from pynasonde.vipir.ngi.plotlib import Ionogram
    from pynasonde.vipir.riq.parsers.read_riq import (
        VIPIR_VERSION_MAP,
        RiqDataset,
        adaptive_gain_filter,
    )

    data_file = EXAMPLES_ROOT / "data" / "PL407_2024058061501.RIQ"
    riq = RiqDataset.create_from_file(
        str(data_file),
        unicode="latin-1",
        vipir_config=VIPIR_VERSION_MAP.configs[0],
    )
    ion = adaptive_gain_filter(
        riq.get_ionogram(threshold=50, remove_baseline_noise=True),
        apply_median_filter=True,
        median_filter_size=3,
    )
    assert np.isfinite(ion.powerdB).any()

    ion.powerdB[np.isnan(ion.powerdB)] = 0.0

    plot = Ionogram(ncols=1, nrows=1)
    plot.add_ionogram(
        frequency=ion.frequency,
        height=ion.height,
        value=ion.powerdB,
        mode="O/X",
        xlabel="Frequency, MHz",
        ylabel="Virtual Height, km",
        ylim=[70, 1000],
        xlim=[1.8, 22],
        add_cbar=True,
        cbar_label="Power, dB",
        prange=[0, 70],
        del_ticks=False,
    )
    output = tmp_path / "ionogram.png"
    plot.save(output)
    plot.close()

    assert output.exists() and output.stat().st_size > 0
