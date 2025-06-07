import datetime as dt
import json
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SimulationOutputs:
    h: np.array = None
    fh: np.array = None
    tf_sweeps: np.array = None
    h_virtual: np.array = None
    h_virtual_e_model: np.array = None
    h_virtual_e_obs: np.array = None
    tf_sweeps_e: np.array = None
    hv_err: float = np.nan

    def compute_rMdse(self):
        self.hv_err = np.sqrt(
            np.nanmedian((self.h_virtual_e_model - self.h_virtual_e_obs) ** 2)
        )
        return self.hv_err

    @staticmethod
    def merge_results(Obj1, Obj2):
        SimulationOutputs(
            h=Obj1.h, fh=Obj1.fh + Obj1.fh, hv_err=Obj1.hv_err + Obj2.hv_err
        )
        return

    def to_csv(self, fname: str):
        o = pd.DataFrame()
        o["plasma_freq"], o["heights"] = self.fh, self.h
        o.to_csv(fname, float_format="%g", index=False, header=True)
        return


@dataclass
class TraceEvent:
    description: str = ""
    layer: str = ""
    fv: np.array = None
    ht: np.array = None
    qq: np.array = None
    ndim: int = 399


@dataclass
class Trace:
    filename: str = ""
    date: dt.datetime = None
    events: List[TraceEvent] = None

    @staticmethod
    def load_file(fin: str):
        e = Trace(filename=fin)
        with open(fin, "r") as file:
            data = json.load(file)
            e.date = data["date"] if data["date"] else dt.datetime.now()
            e.events = [
                TraceEvent(
                    description=ex["event_description"],
                    fv=np.array(ex["fv"]),  # in MHz
                    ht=np.array(ex["ht"]) * 1e-2,  # in km
                    qq=np.empty(50) * 0,
                )
                for ex in data["events"]
            ]
        return e

    @staticmethod
    def load_xml_sao_file(fin: str, ret_scale: bool = False):
        from pynasonde.digisonde.parsers.sao import SaoExtractor

        ext = SaoExtractor(fin, True, True)
        ext.extract_xml()
        entries = []
        for sao_record in ext.sao.SAORecord:
            e = Trace(filename=fin, date=ext.date, events=[])
            for trace in sao_record.TraceList.Trace:
                e.events.append(
                    TraceEvent(
                        description=trace.Layer + "/" + trace.Polarization,
                        layer=trace.Layer,
                        fv=np.array(trace.FrequencyList),  # in MHz
                        ht=np.array(trace.RangeList),  # in km
                        qq=np.empty(50) * 0,
                    )
                )
            entries.append(e)
        if ret_scale:
            scale = ext.get_scaled_datasets_xml(
                params=["foE", "foF1", "foF2", "hmE", "hmF1", "hmF2", "B0", "B1"]
            )
            return (entries, scale)
        else:
            return entries

    def draw_trace(self, ax):
        logger.info(f"Drawing traces....")
        fv, ht = (
            [x for e in self.events for x in e.fv.tolist()],
            [x for e in self.events for x in e.ht.tolist()],
        )
        fv, ht = np.array(fv).ravel(), np.array(ht).ravel()
        ax.scatter(fv, ht, color="r", marker="s", s=0.5, alpha=0.8)
        ax.set_ylabel("Virtual Height (km)")
        ax.set_xlabel("Frequency (MHz)")
        return


if __name__ == "__main__":
    file_path = "tmp/20250527/KW009_2025147120000_SAO.XML"
    e = Trace.load_xml_sao_file(file_path)
    print(e)
