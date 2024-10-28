from typing import List, Tuple

import cv2
import numpy as np
from loguru import logger

import pynasonde.ngi.utils as utils
from pynasonde.ngi.plotlib import Ionogram
from pynasonde.ngi.source import Dataset


class NoiseProfile(object):

    def __init__(self, type="exp", constant=1.5):
        self.type = type
        self.profile = constant
        return

    def get_exp_profile(self, x: np.array, a0: float, b0: float, x0: float):
        self.profile = a0 * np.exp(-b0 * x / x0)
        return self.profile


class AutoScaler(object):

    def __init__(
        self,
        ds: Dataset,
        noise_profile: NoiseProfile = NoiseProfile(),
        mode: str = "O",
        filter: dict = dict(
            frequency=[0, 10],
            height=[50, 400],
        ),
        apply_filter: bool = True,
        segmentation_method: str = "k-means",
    ):
        self.ds = ds
        self.noise_profile = noise_profile
        self.mode = mode
        self.apply_filter = apply_filter
        self.filter = filter
        self.segmentation_method = segmentation_method
        self.extract()
        return

    def extract(self, mode: str = None):
        mode = mode if mode else self.mode
        logger.info(f"Run {mode}-Mode Scaler")
        self.param_name = f"{mode}_mode_power"
        self.param_noise_name = f"{mode}_mode_noise"
        self.noise_profile = self.noise_profile.profile
        self.noise = np.copy(
            getattr(self.ds, self.param_noise_name) * self.noise_profile
        )
        self.image2d = np.copy(getattr(self.ds, self.param_name))
        self.frequency, self.height = (
            np.copy(self.ds.Frequency / 1e3),
            np.copy(self.ds.Range),
        )
        if self.apply_filter:
            self.image2d[self.image2d < self.noise[:, np.newaxis]] = 0
            self.image2d[
                :,
                (self.height <= self.filter["height"][0])
                | (self.height >= self.filter["height"][1]),
            ] = 0
            self.image2d[
                (self.frequency <= self.filter["frequency"][0])
                | (self.frequency >= self.filter["frequency"][1]),
                :,
            ] = 0
        return

    def image_segmentation(self, segmentation_method: str = None, **kwargs):
        segmentation_method = (
            segmentation_method if segmentation_method else self.segmentation_method
        )
        logger.info(f"Running {segmentation_method} image segmentation...")
        if segmentation_method == "k-means":
            self.kmeans_image_segmentation(np.copy(self.image2d), *kwargs)
        return

    def kmeans_image_segmentation(
        self,
        image: np.array,
        K: int = 3,
        criteria: Tuple = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1),
        center_selection: int = cv2.KMEANS_PP_CENTERS,
    ):
        pixels = np.float32(image)
        pixels = pixels.reshape((-1, 3))
        attempts = criteria[1]
        self.compactness, self.label, self.center = cv2.kmeans(
            pixels,
            K,
            None,
            criteria,
            attempts,
            center_selection,
        )
        self.center = np.uint8(self.center)
        self.segmented_image = self.center[self.label.flatten()]
        self.segmented_image = self.segmented_image.reshape(image.shape)
        return

    def to_binary_traces(
        self,
        nbins: int = 1000,
        thresh: float = 2,
        eps: int = 7,
        min_samples: int = 40,
        trace_params: dict = dict(
            region_thick_thresh=100,
        ),
    ):
        import pandas as pd
        from skimage.filters import threshold_otsu
        from sklearn.cluster import DBSCAN

        thresh = thresh * threshold_otsu(np.copy(self.image2d), nbins=nbins)
        logger.info(f"Otsu's thresh {thresh}")
        self.binary_image = self.segmented_image > thresh

        vertices = np.where(self.binary_image == True)
        self.indices, self.traces, self.trace_params = [], dict(), dict()
        for i, j in zip(vertices[0], vertices[1]):
            self.indices.append(
                dict(frequency=self.frequency[i], height=self.height[j])
            )
        self.indices = pd.DataFrame.from_records(self.indices)
        self.indices.dropna(inplace=True)

        if len(self.indices) > 0:
            # Run DBScan to identify individual regions
            dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(
                self.indices[["frequency", "height"]]
            )
            self.indices["labels"] = dbscan.labels_
            self.indices = self.indices[self.indices.labels != -1]

            for label in np.unique(self.indices["labels"]):
                trace = self.indices[self.indices.labels == label]
                if (
                    trace.height.max() - trace.height.min()
                    > trace_params["region_thick_thresh"]
                ):
                    logger.info("Identified region is too thick")
                    # TODO
                    u_freq = np.unique(trace.frequency)
                    for f_bin in zip(u_freq[:-1], u_freq[1:]):
                        print(f_bin)

                self.traces[label] = trace.copy()
                self.trace_params[label] = {
                    "hs": trace.height.min(),
                    "fs": trace.frequency.max(),
                }
            if len(self.traces) > 0:
                self.ds.set_traces(self.traces, self.trace_params)
        return

    def draw_sanity_check_images(
        self,
        fname: str,
        cmap: str = "Greens",
        xticks: List[float] = [1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],
        xlabel: str = "Frequency, MHz",
        ylabel: str = "Virtual Height, km",
        prange: List[float] = [5, 70],
        ylim: List[float] = [50, 800],
        xlim: List[float] = [1, 22],
        font_size: float = 10,
        figsize: tuple = (6, 3),
    ):
        ion = Ionogram(
            fig_title=f"{self.ds.StationName.strip()} / {self.ds.time.strftime('%H:%M:%S UT %d %b %Y')} / {self.mode}-Mode",
            font_size=font_size,
            figsize=figsize,
        )
        ion.add_ionogram(
            self.frequency,
            self.height,
            getattr(self.ds, self.param_name),
            mode=self.mode,
            text="(a) Raw ionogram",
            xlabel="",
            ylabel="",
            cmap=cmap,
            ylim=ylim,
            xlim=xlim,
            prange=prange,
            xticks=xticks,
        )
        ion.add_ionogram(
            self.frequency,
            self.height,
            self.image2d,
            mode=self.mode,
            text="(b) Filtered ionogram",
            xlabel="",
            ylabel="",
            cmap=cmap,
            ylim=ylim,
            xlim=xlim,
            prange=prange,
            xticks=xticks,
        )
        ion.add_ionogram(
            self.frequency,
            self.height,
            self.segmented_image,
            mode=self.mode,
            text="(c) Segmented ionogram",
            xlabel="",
            ylabel="",
            cmap=cmap,
            ylim=ylim,
            xlim=xlim,
            prange=prange,
            xticks=xticks,
        )
        ion.add_ionogram(
            self.frequency,
            self.height,
            self.binary_image,
            mode=self.mode,
            text="(d) Binary ionogram",
            xlabel=xlabel,
            ylabel=ylabel,
            cmap=cmap,
            ylim=ylim,
            xlim=xlim,
            xticks=xticks,
            del_ticks=False,
            prange=[0, 1],
        )

        ax = ion.add_ionogram(
            self.frequency,
            self.height,
            self.binary_image,
            mode=self.mode,
            text="(e) Trace",
            xlabel="",
            ylabel="",
            cmap=cmap,
            ylim=ylim,
            xlim=xlim,
            xticks=xticks,
            prange=[0, 1],
        )
        for key in list(self.traces.keys()):
            trace = self.traces[key]
            ax.plot(
                np.log10(trace["frequency"]),
                trace["height"],
                marker="o",
                ls="None",
                color=utils.get_color_by_index(key, len(self.traces.keys())),
                ms=0.3,
                zorder=5,
                alpha=0.6,
            )

        ax = ion.add_ionogram(
            self.frequency,
            self.height,
            self.binary_image,
            mode=self.mode,
            text="(f) Scaled",
            xlabel="",
            ylabel="",
            cmap=cmap,
            ylim=ylim,
            xlim=xlim,
            xticks=xticks,
            prange=[0, 1],
        )
        for key in list(self.trace_params.keys()):
            trace = self.trace_params[key]
            color = utils.get_color_by_index(key, len(self.traces.keys()))
            ax.axvline(
                np.log10(trace["fs"]), color=color, ls="--", lw=0.6, alpha=0.8, zorder=5
            )
            ax.axhline(trace["hs"], color=color, ls="--", lw=0.6, alpha=0.8, zorder=5)
            trace = self.traces[key]
            ax.plot(
                np.log10(trace["frequency"]),
                trace["height"],
                marker="o",
                ls="None",
                color=utils.get_color_by_index(key, len(self.traces.keys())),
                ms=0.3,
                zorder=5,
                alpha=0.6,
            )
        ion.save(fname)
        ion.close()
        logger.info(f"Save file to: {fname}")
        return
