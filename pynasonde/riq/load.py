from dataclasses import dataclass
from typing import List

from pynasonde.riq.headers.pct import PctType
from pynasonde.riq.headers.pri import PriType
from pynasonde.riq.headers.sct import SctType


@dataclass
class RiqDataset:
    fname: str
    sct: SctType
    pcts: List[PctType]
    pris: List[PriType]

    @classmethod
    def create_from_file(cls, fname: str):
        riq = cls()
        riq.sct, riq.pcts, riq.pris = SctType(), [], []
        riq.sct.read_sct(fname)
        for j in range(riq.sct.timing.pri_count):
            pct = PctType()
            pct.load_sct(riq.sct)
            riq.pcts.append(pct)
        return riq
