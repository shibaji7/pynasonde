from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import numpy as np
from loguru import logger

from pynasonde.vipir.riq.headers.pct import PctType
from pynasonde.vipir.riq.headers.pri import PriType
from pynasonde.vipir.riq.headers.sct import SctType

# from pynasonde.vipir.riq.trace import extract_echo_traces

# Define a mapping for VIPIR version configurations
VIPIR_VERSION_MAP = SimpleNamespace(
    **dict(
        One=dict(
            vipir_version=0,  # Version identifier
            value_Size=4,  # Size of values in bytes
            np_format="int32",  # NumPy data type format
            swap=True,  # Whether to swap byte order
        ),
        Two=dict(
            vipir_version=1,  # Version identifier
            value_Size=2,  # Size of values in bytes
            np_format="int16",  # NumPy data type format
            swap=False,  # Whether to swap byte order
        ),
    )
)


@dataclass
class IonogramDataset:
    """
    Represents an ionogram with frequency, range gates, and amplitude data.

    Attributes:
        frequencies (np.ndarray): Frequencies in MHz.
        range_gates (np.ndarray): Range gates in km.
        amplitude (np.ndarray): Amplitude data.
    """

    frequencies: np.ndarray
    range_gates: np.ndarray
    amplitude: np.ndarray


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
        riq.sct, riq.pcts, riq.pris, riq.pulset = SctType(), [], [], []

        # Read SCT (System Configuration Table) data
        riq.sct.read_sct(fname, unicode)

        # Temporary storage for pulse sets
        pset = []

        # Iterate through all PRI (Pulse Repetition Interval) entries
        for j in range(1, riq.sct.timing.pri_count + 1):
            # Create and load PCT (Pulse Configuration Table) data
            pct = PctType()
            pct.load_sct(riq.sct, vipir_version)
            pct.read_pct(fname, j, riq.unicode)
            riq.pcts.append(pct)

            # Create PRI data using PCT and SCT information
            pri = PriType(
                frequency=pct.frequency,  # Frequency of the pulse
                max_rx=pct.num_receivers,  # Maximum number of receivers
                max_rg=pct.num_gates,  # Maximum number of gates
                ut_time=pct.pri_ut,  # Universal time of the PRI
                relative_time=pct.pri_time_offset,  # Relative time offset
                receiver_count=pct.num_receivers,  # Number of receivers
                gate_count=pct.num_gates,  # Number of gates
                pulset_index=pct.pulse_id,  # Pulse ID
                pulset_length=riq.sct.frequency.pulse_count,  # Total pulse count
                gate_start=riq.sct.timing.gate_start * 0.15,  # Start of the gate in km
                gate_end=riq.sct.timing.gate_end * 0.15,  # End of the gate in km
                gate_step=riq.sct.timing.gate_step
                * 0.15,  # Step size of the gate in km
                a_scan=pct.IQRxRG,  # I/Q data for each receiver and range gate
            )
            riq.pris.append(pri)

        # Create pubset based on FrequencyType.tune_type
        logger.info(
            f"FrequencyType.tune_type: {riq.sct.frequency.tune_type}, pulse_count: {riq.sct.frequency.pulse_count}"
        )
        # If tune_type is 1, group pulses into sets of pulse_count
        if riq.sct.frequency.tune_type == 1:
            for j, pri, pct in zip(
                range(1, riq.sct.timing.pri_count + 1), riq.pris, riq.pcts
            ):
                # Add PRI and PCT data to the current pulse set
                pset.append(dict(pri=pri, pct=pct))
                # Group pulses into sets of pulse_count
                if np.mod(j, riq.sct.frequency.pulse_count) == 0:
                    riq.pulset.append(pset)
                    pset = []
        # If tune_type is >=4, group pulses based on special frequency and pulse_count
        if riq.sct.frequency.tune_type >= 4:
            riq.swap_pulset = []
            riq.swap_frequency = riq.sct.frequency.base_table[1]
            for j, pri, pct in zip(
                range(1, riq.sct.timing.pri_count + 1), riq.pris, riq.pcts
            ):
                if pct.frequency == riq.swap_frequency:
                    # Add PRI and PCT data to the current pulse set
                    riq.swap_pulset.append(dict(pri=pri, pct=pct))
                else:
                    # Add PRI and PCT data to the current pulse set
                    pset.append(dict(pri=pri, pct=pct))
                # Group pulses into sets of pulse_count
                if np.mod(j, riq.sct.frequency.pulse_count * 2) == 0:
                    riq.pulset.append(pset)
                    pset = []
            logger.info(
                f"Swap Frequency: {riq.swap_frequency}, Number of swap_pulset: {len(riq.swap_pulset)}"
            )
        # Log the number of pulses and pulse sets
        logger.info(
            f"Number of pulses: {riq.sct.timing.pri_count}, and PRI Count: {riq.sct.timing.pri_count}, Pset Count:{riq.sct.frequency.pulse_count}, Pulset: {len(riq.pulset)}"
        )
        return riq

    def ionogram(self):
        """
        Returns the ionogram data from the dataset.
        """
        # Check if the dataset is empty
        if not self.pcts:
            logger.warning("Empty dataset")
            return None
        logger.info(
            f"This RIQ datasets will produce {self.sct.timing.ionogram_count} ionogram(s)"
        )
        frequencies = self.sct.frequency.base_table
        # Convert frequencies to MHz
        frequencies = np.array(frequencies) / 1e3
        # Limit to only scaned frequencies
        frequencies = frequencies[: self.sct.frequency.base_steps]
        if self.swap_frequency > 0:
            # If swap_frequency is set, use the swapped frequency
            frequencies = frequencies[::2]

        # Locate Range gates in km from us
        range_gates = np.arange(
            self.sct.timing.gate_start * 0.15,
            self.sct.timing.gate_end * 0.15,
            self.sct.timing.gate_step * 0.15,
        )  # Converted range gates to km
        ionogram = np.zeros((len(frequencies), len(range_gates)), dtype=np.float128)

        # Integrate the amplitude for each frequency component
        for i, pset in enumerate(self.pulset):
            # Iterate through each pulse set
            for _, p in enumerate(pset):
                # Extract PRI and PCT data
                pri = p["pri"]
                # Calculate the integrated amplitude for all receivers
                ionogram[i, :] += pri.read_dB_amplitude_for_ionogram()
        ionogram /= self.sct.frequency.pulse_count
        id = IonogramDataset(
            frequencies=frequencies,
            range_gates=range_gates,
            amplitude=ionogram,
        )
        return id
