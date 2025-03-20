import datetime as dt
import json
from dataclasses import dataclass
from typing import List

import numpy as np
from loguru import logger


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


if __name__ == "__main__":
    file_path = "tmp/polan/ionogram_data.json"
    e = ScaledEntries.load_file(file_path)
    print(e)
