from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import numpy as np
from loguru import logger

from pynasonde.riq.headers.pct import PctType
from pynasonde.riq.headers.pri import PriType
from pynasonde.riq.headers.sct import SctType

# Define a mapping for VIPIR version configurations
VIPIR_VERSION_MAP = SimpleNamespace(
    **dict(
        One=dict(
            vipir_version=0,  # Version identifier
            value_Size=4,  # Size of values in bytes
            np_format="int32",  # NumPy data type format
            swap=False,  # Whether to swap byte order
        ),
    )
)


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
            )
            riq.pris.append(pri)

            # Add PRI and PCT data to the current pulse set
            pset.append(dict(pri=pri, pct=pct))

            # Group pulses into sets of 8
            if np.mod(j, 8) == 0:
                riq.pulset.append(pset)
                pset = []

        # Log the number of pulses and pulse sets
        logger.info(
            f"Number of pulses: {riq.sct.timing.pri_count}, and Pulset: {riq.sct.timing.pri_count/8}, {len(riq.pulset)}"
        )

        return riq

    def ionogram(self):

        return
