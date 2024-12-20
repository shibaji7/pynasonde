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
    ):
        self.lats, self.alts, self.lons = (lats, alts, lons)
        self.param = np.zeros((len(self.alts), len(self.lats), len(self.lons)))
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
                self.param[:, i, j] = iriout.edens * 1e-6
        # return density in /cc
        return self.param
