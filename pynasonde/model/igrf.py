import datetime as dt

import igrf
import numpy as np


class IGRF(object):

    def __init__(self, event: dt.datetime):
        self.event = event
        return

    def fetch_dataset(
        self,
        lats: np.array,
        lons: np.array,
        alts: np.array,
    ):
        self.lats, self.alts, self.lons = (lats, alts, lons)
        B_north, B_east, B_down, B_tot, B_incl, B_decl = (
            np.zeros((len(self.alts), len(self.lats), len(self.lons))),
            np.zeros((len(self.alts), len(self.lats), len(self.lons))),
            np.zeros((len(self.alts), len(self.lats), len(self.lons))),
            np.zeros((len(self.alts), len(self.lats), len(self.lons))),
            np.zeros((len(self.alts), len(self.lats), len(self.lons))),
            np.zeros((len(self.alts), len(self.lats), len(self.lons))),
        )
        for i in range(len(self.lats)):
            for j in range(len(self.lons)):
                mag = igrf.igrf(
                    self.event.strftime("%Y-%m-%d"),
                    glat=self.lats[i],
                    glon=self.lons[j],
                    alt_km=self.alts,
                )
                (
                    B_north[:, i, j],
                    B_east[:, i, j],
                    B_down[:, i, j],
                    B_tot[:, i, j],
                    B_incl[:, i, j],
                    B_decl[:, i, j],
                ) = (
                    mag.variables["north"][:],
                    mag.variables["east"][:],
                    mag.variables["down"][:],
                    mag.variables["total"][:],
                    mag.variables["incl"][:],
                    mag.variables["decl"][:],
                )
        return B_north, B_east, B_down, B_tot, B_incl, B_decl
