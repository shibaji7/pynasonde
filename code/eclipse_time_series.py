import datetime as dt

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter

from pynasonde.ngi.plotlib import Ionogram
from pynasonde.ngi.source import Trace
from pynasonde.ngi.utils import (
    TimeZoneConversion,
    remove_outliers,
    running_median,
    setsize,
    smooth,
)

LTC = TimeZoneConversion(
    None,
    37.8815,
    -75.4374,
)


def get_sun_time(utc_day, lat=37.8815, long=-75.4374, timezone_str="America/New_York"):
    from astral import LocationInfo
    from astral.sun import sun

    loc = LocationInfo(timezone=timezone_str, latitude=lat, longitude=long)
    s = sun(loc.observer, date=utc_day, tzinfo=timezone_str)
    return s["sunset"].replace(tzinfo=dt.timezone.utc)


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
    df["local_time"] = LTC.utc_to_local_time(df.time.tolist())
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

xlim = [dt.datetime(2024, 4, 7, 12), dt.datetime(2024, 4, 7, 20)]
O = Trace.load_saved_scaled_parameters(
    f"./tmp/{xlim[0].strftime('%Y%m%d')}/scaled/", mode="O"
)
O.dropna(inplace=True)
xlim = [dt.datetime(2024, 4, 7, 12), dt.datetime(2024, 4, 7, 20)]
O = O[O.time >= xlim[0]]
O8 = Trace.load_saved_scaled_parameters(f"./tmp/20240408/scaled/", mode="O")
O8.dropna(inplace=True)
O9 = Trace.load_saved_scaled_parameters(f"./tmp/20240409/scaled/", mode="O")
O9.dropna(inplace=True)

setsize(15)
ion = Ionogram(nrows=2, ncols=1, figsize=(5, 7), font_size=15)
o7f = O[(O.fs >= 8)]
get_sun_time(o7f.time.tolist()[0])
ax = ion._add_axis(del_ticks=False)
ax.set_xlim(xlim)
ax.set_xlabel("")
ax.set_ylim([7, 13])
ax.set_ylabel(r"$foF_2$, MHz")
# ax.axvline(
#     LTC.utc_to_local_time([dt.datetime(2024, 4, 7, 19, 25)])[0],
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="r",
# )
# ax.axvline(
#     LTC.utc_to_local_time([dt.datetime(2024, 4, 7, 18, 40)])[0],
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="g",
# )
# ax.axvline(
#     LTC.utc_to_local_time([dt.datetime(2024, 4, 7, 20, 28)])[0],
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="b",
# )
# ax.axvline(
#     get_sun_time(dt.datetime(2024, 4, 7, 18, 40)),
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="k",
# )
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 2)))
ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))

print(o7f.head())
ax.plot(
    o7f.local_time.tolist()[::3],
    smooth(np.array(o7f.fs), 21)[::3],
    ls="None",
    marker=".",
    color="m",
    ms=4,
    label="7 April",
)
o8f = O8[(O8.fs >= 7.5)]
o8f = o8f[~((o8f.time <= dt.datetime(2024, 4, 8, 19)) & (o8f.fs <= 10))]
ax.plot(
    o8f.local_time[::3] - dt.timedelta(days=1),
    smooth(np.array(o8f.fs), 21)[::3],
    ls="None",
    marker="+",
    color="k",
    ms=4,
    label="8 April/Eclipse Day",
)
o9f = O9[(O9.fs >= 8)]
ax.plot(
    o9f.local_time[::3] - dt.timedelta(days=2),
    smooth(np.array(o9f.fs), 21)[::3],
    ls="None",
    marker=".",
    color="r",
    ms=4,
    label="9 April",
)
ax.legend(loc=1)
ax = ax.twinx()
ax.plot(olc.local_time - dt.timedelta(days=1), olc.occ, color="b", ls="-", lw=0.6)
ax.set_xlim(xlim)
ax.set_ylim(0, 1)
ax.set_ylabel("Obscuration", fontdict=dict(color="blue"))

ax = ion._add_axis(del_ticks=False)
ax.set_xlim(xlim)
ax.set_xlabel("")
ax.set_ylim([1, 5])
ax.set_ylabel(r"$foE$, MHz")
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 2)))
ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
o7e = O[(O.fs < 5)]
ax.plot(
    o7e.local_time.tolist()[::3],
    running_median(np.array(o7e.fs), 21)[::3],
    ls="None",
    marker=".",
    color="m",
    ms=4,
)
o8e = O8[(O8.fs < 4)]
o8e = o8e[
    ~(
        (o8e.time >= dt.datetime(2024, 4, 8, 19))
        & (
            o8e.time
            <= dt.datetime(
                2024,
                4,
                8,
                20,
            )
        )
        & (o8e.fs >= 3.2)
    )
]
ax.plot(
    o8e.local_time[::3] - dt.timedelta(days=1),
    np.array(o8e.fs)[::3],
    ls="None",
    marker="+",
    color="k",
    ms=4,
)
o9e = O9[(O9.fs < 5)]
ax.plot(
    o9e.local_time[::3] - dt.timedelta(days=2),
    smooth(np.array(o9e.fs), 21)[::3],
    ls="None",
    marker=".",
    color="r",
    ms=4,
    label="9 April",
)
ax.set_xlabel("Time, LT (WI)")
ax = ax.twinx()
ax.plot(olc.local_time - dt.timedelta(days=1), olc.occ, color="b", ls="-", lw=0.6)
ax.set_xlim(xlim)
# ax.axvline(
#     LTC.utc_to_local_time([dt.datetime(2024, 4, 7, 19, 25)])[0],
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="r",
# )
# ax.axvline(
#     LTC.utc_to_local_time([dt.datetime(2024, 4, 7, 18, 40)])[0],
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="g",
# )
# ax.axvline(
#     LTC.utc_to_local_time([dt.datetime(2024, 4, 7, 20, 28)])[0],
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="b",
# )
# ax.axvline(
#     get_sun_time(dt.datetime(2024, 4, 7, 18, 40)),
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="k",
# )
ax.set_ylim(0, 1)
ax.set_ylabel("Obscuration", fontdict=dict(color="blue"))

ion.save(f"tmp/{xlim[0].strftime('%d')}TS.png")


ion = Ionogram(nrows=2, ncols=1, figsize=(5, 7), font_size=15)
ax = ion._add_axis(del_ticks=False)
ax.set_xlim(xlim)
ax.set_xlabel("")
ax.set_ylim([200, 400])
ax.set_ylabel(r"$hmF_2$, km")
# ax.axvline(
#     LTC.utc_to_local_time([dt.datetime(2024, 4, 7, 19, 25)])[0],
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="r",
# )
# ax.axvline(
#     LTC.utc_to_local_time([dt.datetime(2024, 4, 7, 18, 40)])[0],
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="g",
# )
# ax.axvline(
#     LTC.utc_to_local_time([dt.datetime(2024, 4, 7, 20, 28)])[0],
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="b",
# )
# ax.axvline(
#     get_sun_time(dt.datetime(2024, 4, 7, 18, 40)),
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="k",
# )
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 2)))
ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
ax.plot(
    o7f.local_time.tolist()[::3],
    o7f.hs[::3],
    ls="None",
    marker=".",
    color="m",
    ms=4,
    label="7 April",
)
o8f = remove_outliers(o8f, "hs", [0.15, 0.85])
ax.plot(
    o8f.local_time[::3] - dt.timedelta(days=1),
    o8f.hs[::3],
    ls="None",
    marker="+",
    color="k",
    ms=4,
    label="8 April/Eclipse Day",
)
o9f = remove_outliers(o9f, "hs", [0.15, 0.85])
ax.plot(
    o9f.local_time[::3] - dt.timedelta(days=2),
    o9f.hs[::3],
    ls="None",
    marker=".",
    color="r",
    ms=4,
    label="9 April",
)
ax.legend(loc=1)
ax = ax.twinx()
ax.plot(olc.local_time - dt.timedelta(days=1), olc.occ, color="b", ls="-", lw=0.6)
ax.set_xlim(xlim)
ax.set_ylim(0, 1)
ax.set_ylabel("Obscuration", fontdict=dict(color="blue"))

ax = ion._add_axis(del_ticks=False)
ax.set_xlim(xlim)
ax.set_xlabel("")
ax.set_ylim([80, 120])
ax.set_ylabel(r"$hmE$, MHz")
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 4)))
ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 2)))
ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
o7e = O[(O.fs < 5)]
ax.plot(
    o7e.local_time.tolist()[::3],
    o7e.hs[::3],
    ls="None",
    marker=".",
    color="m",
    ms=4,
)
o8e = O8[(O8.fs < 4)]
o8e = o8e[
    ~(
        (o8e.time >= dt.datetime(2024, 4, 8, 19))
        & (
            o8e.time
            <= dt.datetime(
                2024,
                4,
                8,
                20,
            )
        )
        & (o8e.fs >= 3.2)
    )
]
ax.plot(
    o8e.local_time[::3] - dt.timedelta(days=1),
    np.array(o8e.hs)[::3],
    ls="None",
    marker="+",
    color="k",
    ms=4,
)
ax.plot(
    o9e.local_time[::3] - dt.timedelta(days=2),
    o9e.hs[::3],
    ls="None",
    marker=".",
    color="r",
    ms=4,
)
ax.set_xlabel("Time, LT (WI)")
ax = ax.twinx()
ax.plot(olc.local_time - dt.timedelta(days=1), olc.occ, color="b", ls="-", lw=0.6)
ax.set_xlim(xlim)
# ax.axvline(
#     LTC.utc_to_local_time([dt.datetime(2024, 4, 7, 19, 25)])[0],
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="r",
# )
# ax.axvline(
#     LTC.utc_to_local_time([dt.datetime(2024, 4, 7, 18, 40)])[0],
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="g",
# )
# ax.axvline(
#     LTC.utc_to_local_time([dt.datetime(2024, 4, 7, 20, 28)])[0],
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="b",
# )
# ax.axvline(
#     get_sun_time(dt.datetime(2024, 4, 7, 18, 40)),
#     ls="--",
#     lw=0.6,
#     alpha=1,
#     color="k",
# )
ax.set_ylim(0, 1)
ax.set_ylabel("Obscuration", fontdict=dict(color="blue"))

ion.save(f"tmp/{xlim[0].strftime('%d')}hS.png")
