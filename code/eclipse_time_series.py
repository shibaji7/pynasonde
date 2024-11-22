import datetime as dt

import matplotlib.dates as mdates
import pandas as pd

from pynasonde.ngi.plotlib import Ionogram
from pynasonde.ngi.source import Trace
from pynasonde.ngi.utils import setsize


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


df = load_file("tmp/WI937_2024098_foF2.dat", dt.datetime(2024, 4, 7))
df8 = load_file("tmp/WI937_2024099_foF2.dat", dt.datetime(2024, 4, 8))
olc = load_occultation("tmp/APEP_WFF_Occultation.txt", dt.datetime(2024, 4, 8))
print(df.head())

xlim = [dt.datetime(2024, 4, 7, 16), dt.datetime(2024, 4, 8)]
O = Trace.load_saved_scaled_parameters(
    f"./tmp/{xlim[0].strftime('%Y%m%d')}/scaled/", mode="O"
)
O.dropna(inplace=True)
xlim = [dt.datetime(2024, 4, 7, 16), dt.datetime(2024, 4, 8)]
O = O[O.time>=xlim[0]]
O8 = Trace.load_saved_scaled_parameters(
    f"./tmp/20240408/scaled/", mode="O"
)
O8.dropna(inplace=True)
# X = Trace.load_saved_scaled_parameters("./tmp/20240408/scaled/", "X")


# dbscan = DBSCAN(eps=4, min_samples=20).fit(O[["fs", "hs", "sza"]])

# O["labels"] = dbscan.labels_
# O = O[O != -1]
setsize(15)
ion = Ionogram(nrows=2, ncols=1, figsize=(5,7), font_size=15)
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
o = O[
    (O.fs>=8)
]
ax = ion.add_TS(
    o.time.tolist(),
    o.fs + 0.6,
    xlim=xlim,
    major_locator=mdates.HourLocator(byhour=range(0, 24, 3)),
    minor_locator=mdates.HourLocator(byhour=range(0, 24, 1)),
    ylabel=r"$foF_2$, MHz", xlabel="",
    ylim=[7, 13],
)
ax.plot(df.time, df.foF2, "m+", ms=4, ls="None", alpha=0.6, label="7 April")
o = O8[
    (O8.fs>=7.5)
]
o = o[
    ~((o.time<=dt.datetime(2024, 4, 8, 19))
    & (o.fs<=10))
]
ax.plot(df8.time-dt.timedelta(days=1), df8.foF2, "b+", ms=4, ls="None", alpha=0.6, label="8 April/Eclipse Day")
ax.plot(o.time-dt.timedelta(days=1), o.fs+0.6, "b.", ms=0.7, ls="None")
ax.legend(loc=1)
ax = ax.twinx()
ax.plot(olc.time-dt.timedelta(days=1), olc.occ, color="k", ls="-", lw=0.8)
ax.set_xlim(xlim)
ax.set_ylim(0, 1)
ax.set_ylabel("Obscuration")
o = O[
    (O.fs>=8)
]
ax = ion.add_TS(
    o.time.tolist(),
    o.hs,
    xlim=xlim,
    ylim=[150, 350],
    ylabel=r"$hm_s$, km",
    color="r",
    major_locator=mdates.HourLocator(byhour=range(0, 24, 3)),
    minor_locator=mdates.HourLocator(byhour=range(0, 24, 1)),
)
o = O8[
    (O8.fs>=7.5)
]
o = o[
    ~((o.time<=dt.datetime(2024, 4, 8, 19))
    & (o.fs<=10))
]
ax.plot(o.time-dt.timedelta(days=1), o.hs, "b.", ms=0.7, ls="None")
ax = ax.twinx()
ax.plot(olc.time-dt.timedelta(days=1), olc.occ, color="k", ls="-", lw=0.8)
ax.set_xlim(xlim)
ax.set_ylim(0, 1)
ax.set_ylabel("Obscuration")
ion.save(f"tmp/{xlim[0].strftime('%d')}TS.png")
