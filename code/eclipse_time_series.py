import datetime as dt

from pynasonde.ngi.plotlib import Ionogram
from pynasonde.ngi.source import Trace

O = Trace.load_saved_scaled_parameters("./tmp/20240408/scaled/")
# X = Trace.load_saved_scaled_parameters("./tmp/20240408/scaled/", "X")


# dbscan = DBSCAN(eps=4, min_samples=20).fit(O[["fs", "hs", "sza"]])

# O["labels"] = dbscan.labels_
# O = O[O != -1]
ion = Ionogram(nrows=2, ncols=1, figsize=(5, 8))
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
ion.add_TS(
    O.time.tolist(),
    O.fs,
    xlim=[dt.datetime(2024, 4, 8), dt.datetime(2024, 4, 8, 3)],
    xlabel="",
)
ion.add_TS(
    O.time.tolist(),
    O.hs,
    xlim=[dt.datetime(2024, 4, 8), dt.datetime(2024, 4, 8, 3)],
    ylabel=r"$hmF_2$, km",
    color="b",
)
ion.save("tmp/08TS.png")
