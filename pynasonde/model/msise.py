import datetime as dt

import numpy as np
from nrlmsise00.dataset import msise_4d


class MSISE(object):

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
        self.ds = msise_4d(self.event, alts, lats, lons)
        nn = (
            self.ds.variables["He"][0, :, :, :]
            + self.ds.variables["H"][0, :, :, :]
            + self.ds.variables["O"][0, :, :, :]
            + self.ds.variables["Ar"][0, :, :, :]
            + self.ds.variables["N"][0, :, :, :]
            + self.ds.variables["N2"][0, :, :, :]
            + self.ds.variables["O2"][0, :, :, :]
            + self.ds.variables["rho"][0, :, :, :]
            + self.ds.variables["AnomO"][0, :, :, :]
        )
        return (
            self.ds.variables["He"][0, :, :, :],
            self.ds.variables["H"][0, :, :, :],
            self.ds.variables["O"][0, :, :, :],
            self.ds.variables["Ar"][0, :, :, :],
            self.ds.variables["N"][0, :, :, :],
            self.ds.variables["N2"][0, :, :, :],
            self.ds.variables["O2"][0, :, :, :],
            self.ds.variables["rho"][0, :, :, :],
            self.ds.variables["AnomO"][0, :, :, :],
            self.ds.variables["Texo"][0, :, :, :],
            self.ds.variables["Talt"][0, :, :, :],
            self.ds.variables["Ap"],
            self.ds.variables["f107"],
            self.ds.variables["f107a"],
            nn,
        )
