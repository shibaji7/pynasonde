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
        units: float = 1e6,  # to /cm
    ):
        self.lats, self.alts, self.lons = (lats, alts, lons)
        self.ds = msise_4d(self.event, alts, lats, lons)
        nn = (
            self.ds.variables["He"].values[0, :, :, :]
            + self.ds.variables["H"].values[0, :, :, :]
            + self.ds.variables["O"].values[0, :, :, :]
            + self.ds.variables["Ar"].values[0, :, :, :]
            + self.ds.variables["N"].values[0, :, :, :]
            + self.ds.variables["N2"].values[0, :, :, :]
            + self.ds.variables["O2"].values[0, :, :, :]
            + self.ds.variables["rho"].values[0, :, :, :]
            + self.ds.variables["AnomO"].values[0, :, :, :]
        ) * units
        return (
            self.ds.variables["He"].values[0, :, :, :] * units,
            self.ds.variables["H"].values[0, :, :, :] * units,
            self.ds.variables["O"].values[0, :, :, :] * units,
            self.ds.variables["Ar"].values[0, :, :, :] * units,
            self.ds.variables["N"].values[0, :, :, :] * units,
            self.ds.variables["N2"].values[0, :, :, :] * units,
            self.ds.variables["O2"].values[0, :, :, :] * units,
            self.ds.variables["rho"].values[0, :, :, :],
            self.ds.variables["AnomO"].values[0, :, :, :] * units,
            self.ds.variables["Texo"].values[0, :, :, :],
            self.ds.variables["Talt"].values[0, :, :, :],
            self.ds.variables["Ap"].values,
            self.ds.variables["f107"].values,
            self.ds.variables["f107a"].values,
            nn,
        )
