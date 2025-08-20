from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import numpy as np
from loguru import logger

from pynasonde.vipir.riq.datatypes.default_factory import (
    Exciter_default_factory,
    Frequency_default_factory,
    Monitor_default_factory,
    Reciever_default_factory,
    SCT_default_factory,
    Station_default_factory,
    Timing_default_factory,
)
from pynasonde.vipir.riq.datatypes.pct import PctType
from pynasonde.vipir.riq.datatypes.pri import PriType
from pynasonde.vipir.riq.datatypes.sct import SctType

# Define a mapping for VIPIR version configurations
VIPIR_VERSION_MAP = SimpleNamespace(
    **dict(
        One=dict(
            vipir_version=1,  # Version identifier
            data_type=2,  # Data type identifier
            np_format="float64",  # NumPy data type format
            swap=False,  # Whether to swap byte order
        )
    )
)


@dataclass
class Pulset:
    pcts: List[PctType] = None

    def append(self, pct: PctType) -> None:
        self.pcts.append(pct)
        return

    def __init__(self):
        self.pcts = []
        return


@dataclass
class RiqDataset:
    """
    Represents an RQI dataset containing SCT, PCT, PRI, and pulse information.

    Attributes:
        fname (str): The file name of the dataset.
        sct (SctType): SCT (System Configuration Table) data.
        pcts (List[PctType]): List of PCT (Pulse Configuration Table) data.
        pris (List[PriType]): List of PRI (Pulse Repetition Interval) data.
        pulset (List[List]): Grouped pulse data.
        unicode (str): Encoding format for reading the file.
    """

    fname: str
    sct: SctType = None
    pcts: List[PctType] = None
    pris: List[PriType] = None
    pulset: List[List] = None
    swap_frequency: float = 0.0
    swap_pulset: List = None
    unicode: str = None

    @classmethod
    def create_from_file(
        cls, fname: str, unicode="latin-1", vipir_version: dict = VIPIR_VERSION_MAP.One
    ):
        """
        Factory method to create an RiqDataset instance from a file.

        Args:
            fname (str): The file name to load the dataset from.
            unicode (str): Encoding format for reading the file. Default is "latin-1".
            vipir_version (dict): VIPIR version configuration. Default is VIPIR_VERSION_MAP.One.

        Returns:
            RiqDataset: An instance of the RiqDataset class.
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
                    f, riq.sct, vipir_version, unicode
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
        # Log the number of pulses and pulse sets
        logger.info(
            f"Number of pulses: {riq.sct.timing.pri_count}, and PRI Count: {riq.sct.timing.pri_count}, Pset Count:{riq.sct.frequency.pulse_count}, Pulset: {len(riq.pulsets)}"
        )
        return riq

    def get_ionogram(self):
        snr, frequencies, heights = (
            np.zeros((len(self.pulsets), self.sct.timing.gate_count)),
            np.zeros((len(self.pulsets))),
            np.zeros((self.sct.timing.gate_count)),
        )
        pulse_i, pulse_q = (
            np.array([p.pulse_i for p in self.pulses]),
            np.array([p.pulse_q for p in self.pulses]),
        )
        pulse_i, pulse_q = (
            pulse_i.reshape(
                self.sct.frequency.base_steps,
                self.sct.frequency.pulse_count,
                self.sct.timing.gate_count,
                self.sct.station.rx_count,
            ),
            pulse_q.reshape(
                self.sct.frequency.base_steps,
                self.sct.frequency.pulse_count,
                self.sct.timing.gate_count,
                self.sct.station.rx_count,
            ),
        )
        pulse_i, pulse_q = np.mean(pulse_i, axis=(1, 3)), np.mean(pulse_q, axis=(1, 3))
        snr = np.log10(pulse_i**2 + pulse_q**2)
        snr_base = np.nanmean(snr, axis=1)
        snr = snr - snr_base[:, None]
        frequencies = self.sct.frequency.base_table / 1e3
        heights = (
            np.arange(
                self.sct.timing.gate_start,
                self.sct.timing.gate_end,
                self.sct.timing.gate_step,
            )
            * 0.15
        )  # to km
        return (snr, frequencies, heights)


if __name__ == "__main__":
    RiqDataset.create_from_file(
        "/home/chakras4/Research/ERAUCodeBase/readriq-2.08/Justin/PL407_2024058061501.RIQ"
    ).get_ionogram()
