import json
from dataclasses import dataclass
from typing import List

import numpy as np
from loguru import logger


@dataclass
class Datasets:
    fv: List[float] = None
    ht: List[float] = None
    description: str = ""
    start: float = None


@dataclass
class ScaledEvent:
    description: str = ""
    fh: float = None
    dip: float = None
    a_mode: float = None
    valley: int = None
    data_set: List[Datasets] = None


@dataclass
class ScaledEntries:
    filename: str = ""
    date: str = ""
    events: List[ScaledEvent] = None

    @staticmethod
    def load_file(fin: str):
        e = ScaledEntries(filename=fin)
        with open(fin, "r") as file:
            data = json.load(file)
            e.date = data["date"]
            e.events = [
                ScaledEvent(
                    description=ex["event_description"],
                    fh=ex["fh"],
                    dip=ex["dip"],
                    a_mode=ex["a_mode"],
                    valley=ex["valley"],
                    data_set=[
                        Datasets(
                            description=d["description"],
                            start=d["start"],
                            fv=np.array(d["fv"]),  # in MHz
                            ht=np.array(d["ht"]) * 1e-2,  # in km
                        )
                        for d in ex["data_set"]
                    ],
                )
                for ex in data["events"]
            ]
        return e


if __name__ == "__main__":
    file_path = "tmp/polan/ionogram_data.json"
    e = ScaledEntries.load_file(file_path)
