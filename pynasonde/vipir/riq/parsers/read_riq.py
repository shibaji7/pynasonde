"""Helpers and data structures for reading VIPIR RIQ (raw IQ) files.

The routines in this module provide the core logic used to decode raw VIPIR
records into the higher-level pulse and ionogram representations consumed by
the rest of the pipeline.  They also supply utility filters that can be reused
when post-processing ionograms for visualization.
"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import cv2
import numpy as np
from loguru import logger

from pynasonde.vipir.ngi.utils import load_toml
from pynasonde.vipir.riq.datatypes.default_factory import (
    Exciter_default_factory,
    Frequency_default_factory,
    Monitor_default_factory,
    Reciever_default_factory,
    SCT_default_factory,
    Station_default_factory,
    Timing_default_factory,
)
from pynasonde.vipir.riq.datatypes.pct import Ionogram, PctType
from pynasonde.vipir.riq.datatypes.sct import SctType
from pynasonde.vipir.riq.parsers.trace import extract_echo_traces

# Define a mapping for VIPIR version configurations
VIPIR_VERSION_MAP = load_toml().vipir_data_format_maps


def find_thresholds(
    data: np.ndarray, bins: int = 100, prominence: float = 100, **kwargs
):
    """Estimate power thresholds by locating valleys in the histogram.

    The signal is sanitized (NaNs, infs removed), histogrammed, and the
    prominence of the negated counts is inspected.  The first dip is returned
    as a reasonable power threshold for separating noise from echoes.

    Args:
        data: Power (dB) samples to analyze.
        bins: Number of histogram bins.
        prominence: `scipy.signal.find_peaks` prominence setting applied to the
            inverted histogram.
        **kwargs: Forwarded to both `numpy.histogram` and `find_peaks`.

    Returns:
        Tuple of (`dip_bins`, `first_threshold`) where `dip_bins` contains the
        bin edges surrounding each detected dip and `first_threshold` is the
        first element (often used as the working threshold).
    """
    from scipy.signal import find_peaks

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).flatten()
    data = data[data > 0.0]
    counts, bin_edges = np.histogram(data, bins=bins, *kwargs)
    dips, _ = find_peaks(-counts, prominence=prominence, *kwargs)
    dip_bins = bin_edges[dips]
    print(dip_bins)
    return dip_bins, dip_bins[0]


def remove_morphological_noise(
    ion: Ionogram,
    threshold: float = 0.0,
    morf_type=cv2.MORPH_RECT,
    iterations: int = 1,
    kernel_size: tuple = (1, 3),
    parameter: str = "powerdB",
):
    """Morphologically clean low-amplitude noise from an ionogram slice.

    Args:
        ion: Ionogram in-place modified.
        threshold: Minimum value retained in the output.
        morf_type: Structuring element shape passed to OpenCV.
        iterations: Number of open/close passes.
        kernel_size: Structuring element dimensions.
        parameter: Ionogram attribute to process (defaults to ``powerdB``).

    Returns:
        The provided ionogram instance with its `parameter` attribute filtered.
    """
    data = getattr(ion, parameter)
    img_bin = (data > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(morf_type, kernel_size)
    opened = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=iterations)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    data = data * closed
    data[data <= threshold] = np.nan
    setattr(ion, parameter, data)
    return ion


def adaptive_gain_filter(
    ion: Ionogram,
    snr_threshold: float = 0.0,
    generic_filter_size: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    apply_median_filter: bool = False,
    median_filter_size: int = 3,
    parameter: str = "powerdB",
    **kwargs,
):
    """Apply SNR-adaptive gain and optional median filtering to an ionogram.

    Args:
        ion: Ionogram updated in-place.
        snr_threshold: Values at or below this level are suppressed.
        generic_filter_size: Window size provided to `generic_filter`.
        mode: Padding mode (see `scipy.ndimage.generic_filter`).
        cval: Constant padding value when ``mode=='constant'``.
        apply_median_filter: Whether to follow with a median filter.
        median_filter_size: Median filter window size.
        parameter: Ionogram attribute to process.
        **kwargs: Additional arguments forwarded to SciPy filters.

    Returns:
        The supplied ionogram with its `parameter` attribute smoothed.
    """
    from scipy.ndimage import generic_filter, median_filter

    data = getattr(ion, parameter)
    # Compute gain factor: count non-NaN values in each 3x3 box
    data[data <= snr_threshold] = np.nan
    valid_mask = ~np.isnan(data)
    gain = generic_filter(
        valid_mask.astype(float),
        np.nansum,
        size=generic_filter_size,
        mode=mode,
        cval=cval,
        *kwargs,
    )
    # Optionally normalize gain (e.g., divide by n**2 for a nXn box)
    gain = gain / generic_filter_size**2
    data *= gain
    if apply_median_filter:
        data = np.nan_to_num(data, nan=0.0)
        data = median_filter(
            data,
            size=median_filter_size,
            mode=mode,
            cval=cval,
            *kwargs,
        )
    data[data <= snr_threshold] = np.nan
    setattr(ion, parameter, data)
    return ion


@dataclass
class Pulset:
    """Container grouping PCT records that share a pulse definition.

    Attributes:
        pcts: Mutable list of `PctType` entries accumulated for this pulse set.
    """

    pcts: List[PctType] = None

    def append(self, pct: PctType) -> None:
        """Attach a pulse configuration to the group."""
        self.pcts.append(pct)
        return

    def __init__(self):
        self.pcts = []
        return


@dataclass
class RiqDataset:
    """Aggregate view of an RIQ capture (SCT tables, pulses, ionograms).

    Attributes:
        fname: Source filename for the dataset.
        sct: System configuration structure populated from the file.
        pcts: Flat list of all PCT entries parsed from the capture.
        pulset: Sequence of grouped pulse sets honoring tune conditions.
        swap_frequency: Active swap frequency used when `tune_type >= 4`.
        swap_pulset: Collected pulses that match the swap frequency.
        unicode: Encoding used for textual fields.
    """

    fname: str
    sct: SctType = None
    pcts: List[PctType] = None
    pulset: List[List] = None
    swap_frequency: float = 0.0
    swap_pulset: List = None
    unicode: str = None

    @classmethod
    def create_from_file(
        cls,
        fname: str,
        unicode="latin-1",
        vipir_config: SimpleNamespace = VIPIR_VERSION_MAP.configs[0],
    ):
        """Create a dataset by parsing the given RIQ file.

        Args:
            fname: Path to the RIQ capture.
            unicode: Encoding used when reading text fields.
            vipir_config: Version-specific mapping loaded from TOML metadata.

        Returns:
            RiqDataset: Parsed dataset including SCT and pulse information.
        """
        # Initialize the dataset
        riq = cls(fname)
        riq.unicode = unicode
        riq.sct, riq.pulses, riq.pulsets = SctType(), [], []

        with open(fname, mode="rb") as f:
            # Read SCT (System Configuration Table) data
            riq.sct.read_sct_from_file_pointer(f, unicode)
            # Read SCT.Station Data
            riq.sct.station.read_station_from_file_pointer(f, unicode)
            # Read SCT.Timing Data
            riq.sct.timing.read_timing_from_file_pointer(f, unicode)
            # Read SCT.Frequency Data
            riq.sct.frequency.read_frequency_from_file_pointer(f, unicode)
            # Read SCT.Reciever Data
            riq.sct.receiver.read_reciever_from_file_pointer(f, unicode)
            # Read SCT.Exciter Data
            riq.sct.exciter.read_exciter_from_file_pointer(f, unicode)
            # Read SCT.Monitor Data
            riq.sct.monitor.read_monitor_from_file_pointer(f, unicode)
            # Fix all SCT strings
            riq.sct.fix_SCT_strings()

            # Load all PRI, PCT, and pulse data
            for j in range(1, riq.sct.timing.pri_count + 1):
                # Create and load PCT (Pulse Configuration Table) data
                pct = PctType().read_pct_from_file_pointer(
                    f, riq.sct, vipir_config, unicode
                )
                riq.pulses.append(pct)

        # If tune_type is 1, group pulses into sets of pulse_count
        if riq.sct.frequency.tune_type == 1:
            pulset = Pulset()
            for j, pulse in zip(range(1, riq.sct.timing.pri_count + 1), riq.pulses):
                # Add PCT data to the current pulse set
                pulset.append(pulse)
                # Group pulses into sets of pulse_count
                if np.mod(j, riq.sct.frequency.pulse_count) == 0:
                    riq.pulsets.append(pulset)
                    pulset = Pulset()
        # If tune_type is >=4, group pulses based on special frequency and pulse_count
        elif riq.sct.frequency.tune_type >= 4:
            riq.swap_pulsets = []
            riq.swap_frequency = riq.sct.frequency.base_table[1]
            pulset = Pulset()
            for j, pulse in zip(range(1, riq.sct.timing.pri_count + 1), riq.pulses):
                if pulse.frequency == riq.swap_frequency:
                    # Add PRI and PCT data to the current pulse set
                    riq.swap_pulsets.append(pulse)
                else:
                    # Add PRI and PCT data to the current pulse set
                    pulset.append(pulse)
                # Group pulses into sets of pulse_count
                if np.mod(j, riq.sct.frequency.pulse_count * 2) == 0:
                    riq.pulsets.append(pulset)
                    pulset = Pulset()
            logger.info(
                f"Swap Frequency: {riq.swap_frequency}, Number of swap_pulsets: {len(riq.swap_pulsets)}"
            )
        else:
            raise NotImplementedError(
                f"tune_type {riq.sct.frequency.tune_type} not implemented"
            )
        # Log the number of pulses and pulse sets
        logger.info(
            f"Number of pulses: {riq.sct.timing.pri_count}, and PRI Count: {riq.sct.timing.pri_count}, Pset Count:{riq.sct.frequency.pulse_count}, Pulset: {len(riq.pulsets)}"
        )
        return riq

    def get_ionogram(
        self,
        threshold: float = None,
        remove_baseline_noise: bool = False,
        bins: int = 100,
        prominence: float = 100,
        **kwargs,
    ) -> Ionogram:
        """Convert decoded pulses into an averaged ionogram.

        Args:
            threshold: Optional lower bound applied to the resulting power (dB).
            remove_baseline_noise: Whether to subtract the median baseline.
            bins: Histogram bins for automatic thresholding.
            prominence: Prominence value used by `find_thresholds`.
            **kwargs: Extra options forwarded to `find_thresholds`.

        Returns:
            Ionogram constructed from the dataset's pulse data.
        """
        ion = Ionogram()
        ion.frequency = (
            np.array([psets.pcts[0].frequency for psets in self.pulsets]) / 1e3
        )
        ion.height = (
            np.arange(
                self.sct.timing.gate_start,
                self.sct.timing.gate_end,
                self.sct.timing.gate_step,
            )
            * 0.15
        )  # to km
        pulse_i, pulse_q = (
            np.array([p.pulse_i for psets in self.pulsets for p in psets.pcts]),
            np.array([p.pulse_q for psets in self.pulsets for p in psets.pcts]),
        )
        extract_echo_traces(self.sct, pulse_i, pulse_q)
        pulse_i, pulse_q = (
            pulse_i.reshape(
                len(self.pulsets),
                self.sct.frequency.pulse_count,
                self.sct.timing.gate_count,
                self.sct.station.rx_count,
            ),
            pulse_q.reshape(
                len(self.pulsets),
                self.sct.frequency.pulse_count,
                self.sct.timing.gate_count,
                self.sct.station.rx_count,
            ),
        )
        ion.pulse_i, ion.pulse_q = np.mean(pulse_i, axis=(1, 3)), np.mean(
            pulse_q, axis=(1, 3)
        )
        # Calculate power, power base, and SNR
        power = np.sqrt(ion.pulse_i**2 + ion.pulse_q**2)
        ion.power = (
            power - np.median(power, axis=1)[:, None]
            if remove_baseline_noise
            else power
        )
        ion.powerdB = 10 * np.log10(ion.power)
        if threshold is None:
            _, threshold = find_thresholds(
                ion.powerdB, bins=bins, prominence=prominence, *kwargs
            )
        logger.info(f"Threshold: {threshold}")
        ion.powerdB[ion.powerdB < threshold] = 0.0
        # Calculate phase
        # ion.phase = np.arctan2(ion.pulse_i, ion.pulse_q)
        # print(ion.phase)
        # # ion.phase = np.unwrap(ion.phase, axis=1)
        return ion


if __name__ == "__main__":
    RiqDataset.create_from_file(
        # "/home/chakras4/Research/ERAUCodeBase/readriq-2.08/Justin/PL407_2024058061501.RIQ"
        "tmp/WI937_2022233235902.RIQ",
        vipir_config=VIPIR_VERSION_MAP.configs[1],
    )  # .get_ionogram()
