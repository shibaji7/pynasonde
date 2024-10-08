import numpy as np

from pynasonde.ngi.ionograms import DataSource
from pynasonde.ngi.utils import get_gridded_parameters


def smooth(x, window_len=11, window="hanning"):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ["flat", "hanning", "hamming", "bartlett", "blackman"]:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        )
    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-2 : -window_len - 1 : -1]]
    if window == "flat":
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")
    y = np.convolve(w / w.sum(), s, mode="valid")
    d = window_len - 1
    y = y[int(d / 2) : -int(d / 2)]
    return y


def scan_method(source, idx, mode: str = "O", noise_scale: float = 1.3):
    import pandas as pd

    Zval = np.copy(getattr(source, f"{mode}_mode_power"))
    frequency, rrange = np.copy(source.Frequency / 1e3), np.copy(source.Range)
    noise_level = np.copy(getattr(source, f"{mode}_mode_noise"))
    # Filter high frequency
    Zval[frequency > 10, :] = np.nan
    # Filter D-F regions
    Zval[:, (rrange <= 50) | (rrange >= 400)] = np.nan
    # Remove noises
    Zval[Zval < noise_level[:, np.newaxis] * noise_scale] = np.nan
    frequency, rrange = np.meshgrid(frequency, rrange)
    o = pd.DataFrame()
    o["freq"] = frequency.ravel()
    o["range"] = rrange.ravel()
    o["val"] = Zval.T.ravel()
    o.dropna(inplace=True)
    # print(Zval.shape,frequency.shape, rrange.shape)
    print(o.head())
    from sklearn.cluster import DBSCAN

    clustering = DBSCAN(eps=3, min_samples=20).fit(o[["freq", "range", "val"]].values)
    o["labels"] = clustering.labels_
    o = o[(o["labels"] != -1) & (o["labels"] == o.labels.value_counts().argmax())]
    print(np.unique(clustering.labels_))
    print(o.head())

    X, Y, Z = get_gridded_parameters(o, "freq", "range", "val")
    print(X[0, :], Y[:, 0], Z.shape)
    ff, hh = X[0, :], Y[:, 0]

    indices = np.where(~np.isnan(Z))
    # print(indices)
    trace = dict(frequency=[], range=[])
    for i, j in zip(indices[0], indices[1]):
        trace["frequency"].append(ff[i])
        trace["range"].append(hh[j])
    trace["range"] = smooth(
        np.array(trace["range"]), min(71, int(len(trace["range"]) / 2) * 2 - 11)
    )
    setattr(source, "trace", trace)

    setattr(source, "foF2", 0.834 * np.max(trace["frequency"]))
    setattr(
        source,
        "hmF2",
        trace["range"][
            np.argmin(
                np.abs(0.834**2 * np.max(trace["frequency"]) - trace["frequency"])
            )
        ],
    )

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4.5, 3 * 1), dpi=300)
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(
        X,
        Y,
        Z.T,
        lw=0.01,
        edgecolors="None",
        cmap="Greens",
        vmax=0,
        vmin=100,
        zorder=3,
    )
    ax.set_ylim(50, 800)
    ax.set_xlim(1, 22)
    pos = ax.get_position()
    cpos = [
        pos.x1 + 0.025,
        pos.y0 + 0.0125,
        0.015,
        pos.height * 0.9,
    ]  # this list defines (left, bottom, width, height
    cax = fig.add_axes(cpos)
    cb = fig.colorbar(im, ax=ax, cax=cax)
    cb.set_label("O Mode")

    ax.plot(
        source.trace["frequency"],
        source.trace["range"],
        "k-",
        lw=0.5,
        alpha=0.8,
        zorder=5,
    )
    ax.axvline(source.foF2, ls="--", color="b", lw=0.4)
    ax.axhline(source.hmF2, ls="--", color="r", lw=0.4)

    fig.savefig(f"tmp/scan{idx}.png")

    return


# Projection Method for Shape Analysis
def projection_method(source, k: float = 1.5, mode: str = "O", scale=1.5):
    # Extract all the needed parameters
    Zval = np.copy(getattr(source, f"{mode}_mode_power"))
    frequency, rrange = np.copy(source.Frequency / 1e3), np.copy(source.Range)
    noise_level = scale * np.copy(getattr(source, f"{mode}_mode_noise"))  # Noise

    # Filter high frequency
    Zval[frequency > 10, :] = np.nan
    # Filter D-F regions
    Zval[:, (rrange <= 50) | (rrange >= 400)] = np.nan
    # Remove noises
    Zval[Zval < noise_level[:, np.newaxis]] = np.nan
    setattr(source, f"{mode}_mode_power", Zval)

    # Extract all the trace parameters
    indices = np.where(~np.isnan(Zval))
    trace = dict(frequency=[], range=[])
    for i, j in zip(indices[0], indices[1]):
        trace["frequency"].append(frequency[i])
        trace["range"].append(rrange[j])
    trace["range"] = smooth(
        np.array(trace["range"]), min(151, int(len(trace["range"]) / 2) * 2 - 11)
    )
    setattr(source, "trace", trace)

    setattr(source, "foF2", 0.834 * np.max(trace["frequency"]))
    setattr(
        source,
        "hmF2",
        trace["range"][
            np.argmin(
                np.abs(0.834**2 * np.max(trace["frequency"]) - trace["frequency"])
            )
        ],
    )


ds = DataSource(source_folder="./tmp/20240407/")
ds.load_data_sets(0, 1)
from pynasonde.ngi.scale import AutoScaler

for i, dx in enumerate(ds.datasets):
    a = AutoScaler(dx)
    a.image_segmentation()
    a.to_binary_traces()
    a.draw_sanity_check_images("tmp/scan.png")
# ds.extract_ionograms()
# ds.save_scaled_dataset()
