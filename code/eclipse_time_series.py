import datetime as dt

import matplotlib.dates as mdates
import pandas as pd

from pynasonde.ngi.plotlib import Ionogram
from pynasonde.ngi.source import Trace


def load_occultation(fname, t):
    with open(fname, "r") as f:
        lines = f.readlines()
    df = []
    for line in lines:
        line = list(filter(None, line.replace("\n", "").replace(",", "").split(" ")))
        df.append(
            dict(
                time=t
                + dt.timedelta(hours=float(line[0].split(":")[0]))
                + dt.timedelta(minutes=float(line[0].split(":")[1])),
                occ=float(line[1]),
            )
        )
    df = pd.DataFrame.from_records(df)
    return df


def load_file(fname, t):
    with open(fname, "r") as f:
        lines = f.readlines()
    df = []
    for line in lines:
        line = [float(l) for l in list(filter(None, line.replace("\n", "").split(" ")))]
        df.append(dict(time=t + dt.timedelta(hours=line[0]), foF2=line[1]))
    df = pd.DataFrame.from_records(df)
    return df


df = load_file("tmp/WI937_2024099_foF2.dat", dt.datetime(2024, 4, 8))
olc = load_occultation("tmp/APEP_WFF_Occultation.txt", dt.datetime(2024, 4, 8))
print(df.head())

xlim = [dt.datetime(2024, 4, 8, 16), dt.datetime(2024, 4, 9)]
O = Trace.load_saved_scaled_parameters(
    f"./tmp/{xlim[0].strftime('%Y%m%d')}/scaled/", mode="O"
)
print(O.head())
O.dropna(inplace=True)
# X = Trace.load_saved_scaled_parameters("./tmp/20240408/scaled/", "X")


# dbscan = DBSCAN(eps=4, min_samples=20).fit(O[["fs", "hs", "sza"]])

# O["labels"] = dbscan.labels_
# O = O[O != -1]
ion = Ionogram(nrows=1, ncols=1, figsize=(5, 3))
# r = 0
# for l in np.unique(O.labels):
#     d = O[O.labels == l]
#     print(d.fs.min(), d.fs.max())
#     print(d.hs.min(), d.hs.max())
#     ion.add_TS(
#         d.time.tolist(),
#         d.hs,
#         d.fs,
#         xlim=[dt.datetime(2024, 4, 8), dt.datetime(2024, 4, 8, 3)],
#     )
#     if r == 1:
#         break
#     r += 1
ax = ion.add_TS(
    O.time.tolist(),
    O.fs + 0.6,
    xlim=xlim,
    major_locator=mdates.HourLocator(byhour=range(0, 24, 3)),
    minor_locator=mdates.HourLocator(byhour=range(0, 24, 1)),
    ylabel=r"$fo_s$, MHz",
    ylim=[1, 15],
)
ax.plot(df.time, df.foF2, "ko", ms=1.2, ls="None")
# ion.add_TS(
#     O.time.tolist(),
#     O.hs,
#     xlim=xlim,
#     ylabel=r"$hm_s$, km",
#     color="b",
#     major_locator=mdates.HourLocator(byhour=range(0, 24, 3)),
#     minor_locator=mdates.HourLocator(byhour=range(0, 24, 1)),
# )
ion.save(f"tmp/{xlim[0].strftime('%d')}TS.png")
