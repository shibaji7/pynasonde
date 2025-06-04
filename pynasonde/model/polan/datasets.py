import datetime as dt
import json
from dataclasses import dataclass
from typing import List

import numpy as np
from loguru import logger


@dataclass
class SimulationOutputs:
    h: np.array = None
    fh: np.array = None
    tf_sweeps: np.array = None
    h_virtual: np.array = None


@dataclass
class SimulationDataset:
    fv: np.array = None
    ht: np.array = None
    description: str = ""
    qq: np.array = None
    ndim: int = None


@dataclass
class ScaledEvent:
    description: str = ""
    fv: np.array = None
    ht: np.array = None
    qq: np.array = None
    ndim: int = 399

    def draw_trace(self, ax):
        logger.info(f"Drawing traces....")
        ax.plot(self.fv[: len(self.ht)], self.ht, "r.", ms=0.7, alpha=0.8)
        ax.set_ylabel("Height (km)")
        ax.set_xlabel("Frequency (MHz)")
        return


@dataclass
class ScaledEntries:
    filename: str = ""
    date: dt.datetime = None
    events: List[ScaledEvent] = None

    @staticmethod
    def load_file(fin: str):
        e = ScaledEntries(filename=fin)
        with open(fin, "r") as file:
            data = json.load(file)
            e.date = data["date"] if data["date"] else dt.datetime.now()
            e.events = [
                ScaledEvent(
                    description=ex["event_description"],
                    fv=np.array(ex["fv"]),  # in MHz
                    ht=np.array(ex["ht"]) * 1e-2,  # in km
                    qq=np.empty(50) * 0,
                )
                for ex in data["events"]
            ]
        return e

    @staticmethod
    def load_xml_sao_file(fin: str):
        from pynasonde.digisonde.parsers.sao import SaoExtractor

        ext = SaoExtractor(fin, True, True)
        ext.extract_xml()
        e = ScaledEntries(filename=fin)
        if hasattr(ext.sao.SAORecordList.SAORecord[0], "TraceList"):
            # for _ in range(ext.sao.SAORecordList.SAORecord[0].TraceList[0].Num):

            print(ext.sao.SAORecordList.SAORecord[0].TraceList)
        return e


if __name__ == "__main__":
    file_path = "tmp/20250527/KW009_2025147120000_SAO.XML"
    e = ScaledEntries.load_xml_sao_file(file_path)
    print(e)
