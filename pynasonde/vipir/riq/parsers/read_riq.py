from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import numpy as np
from loguru import logger

from pynasonde.vipir.riq.datatypes.pct import PctType
from pynasonde.vipir.riq.datatypes.pri import PriType
from pynasonde.vipir.riq.datatypes.sct import SctType

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

        # Log the number of pulses and pulse sets
        logger.info(
            f"Number of pulses: {riq.sct.timing.pri_count}, and PRI Count: {riq.sct.timing.pri_count}, Pset Count:{riq.sct.frequency.pulse_count}, Pulset: {len(riq.pulset)}"
        )
        return riq


if __name__ == "__main__":
    RiqDataset.create_from_file("tmp/")