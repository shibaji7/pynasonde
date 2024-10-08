import pandas as pd
import datetime as dt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from pynasonde.ngi.utils import setsize

def load_file(fname, t):
    with open(fname, "r") as f:
        lines = f.readlines()
    df = []
    for line in lines:
        line = [
            float(l)
            for l in list(filter(None, line.replace("\n", "").split(" ")))
        ]
        df.append(dict(
            time = t + dt.timedelta(hours=line[0]),
            foF2 = line[1]
        ))
    df = pd.DataFrame.from_records(df)
    return df

background = load_file("tmp/WI937_2024098_foF2.dat", dt.datetime(2024, 4, 8))
eclipse = load_file("tmp/WI937_2024099_foF2.dat", dt.datetime(2024, 4, 8))

setsize(10)
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(4.5, 3), dpi=300)
ax = fig.add_subplot(111)
hours = mdates.HourLocator(byhour=range(0, 24, 4))
ax.xaxis.set_major_locator(hours)
ax.xaxis.set_major_formatter(DateFormatter(r"%H^{%M}"))
ax.plot(background.time, background.foF2, ls="-", lw=0.9, color="k", label="7 April, 2024")
ax.plot(eclipse.time, eclipse.foF2, ls="-", lw=0.9, color="r", label="8 April, 2024")
ax.legend(loc=2)
ax.axvline(dt.datetime(2024, 4, 8, 19, 25), ls="--", lw=0.7, alpha=0.7, color="g")
ax.axvline(dt.datetime(2024, 4, 8, 18, 40), ls="--", lw=0.7, alpha=0.7, color="g")
ax.axvline(dt.datetime(2024, 4, 8, 20, 28), ls="--", lw=0.7, alpha=0.7, color="g")
ax.set_xlabel("Time, UT", fontdict=dict(size="12"))
ax.set_xlim(dt.datetime(2024, 4, 8), dt.datetime(2024, 4, 9))
ax.set_ylim(5, 15)
ax.set_ylabel(r"$foF_2$, MHz", fontdict=dict(size="12"))
fig.savefig("tmp/foF2.png", bbox_inches="tight")