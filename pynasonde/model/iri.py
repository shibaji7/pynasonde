import datetime as dt

import iricore
import numpy as np


class IRI(object):

    def __init__(self, event: dt.datetime, iri_version: str):
        self.event = event
        self.iri_version = iri_version
        return

    def fetch_dataset(
        self,
        lats: np.array,
        lons: np.array,
        alts: np.array,
        unit: float = 1.0,  # change to 1e-6 if need /cc, as is /cm
    ):
        self.lats, self.alts, self.lons = (lats, alts, lons)
        self.edens, self.itemp, self.etemp = (
            np.zeros((len(self.alts), len(self.lats), len(self.lons))),
            np.zeros((len(self.alts), len(self.lats), len(self.lons))),
            np.zeros((len(self.alts), len(self.lats), len(self.lons))),
        )
        alt_range = [alts[0], alts[-1], alts[1] - alts[0]]
        for i in range(len(self.lats)):
            for j in range(len(self.lons)):
                iriout = iricore.iri(
                    self.event,
                    alt_range,
                    self.lats[i],
                    self.lons[j],
                    self.iri_version,
                )
                self.edens[:, i, j] = iriout.edens * unit
        # return density in /cm, and K
        return self.edens, self.itemp, self.etemp
