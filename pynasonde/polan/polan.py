import datetime as dt
from typing import List, Union

import numpy as np
from loguru import logger

from pynasonde.digisonde.digi_plots import DigiPlots
from pynasonde.model.absorption.constants import pconst
from pynasonde.polan.datasets import ScaledEntries, ScaledEvent, SimulationDataset


class Polan(object):
    """
    ** This is an python implementation of the code in https://github.com/space-physics/POLAN

    A generalised POLynomial real-height ANalysis for ionograms.


    This implements a complex algorithm for ionospheric analysis
    based on polynomial fitting of virtual and real height data. It manages
    different modes of analysis, deals with x-ray data, and can perform
    least-squares fitting for Chapman layer peaks.
    """

    def __init__(
        self,
        entries: ScaledEntries,
        h_max_simulation: float = 500,
        h_steps: Union[float, str] = 1e-3,
    ):
        self.entries = entries
        self.h_max_simulation = h_max_simulation
        self.h_steps = h_steps
        if type(h_steps) == float:
            self.nbins = int(h_max_simulation / h_steps)
        self.polan(self.entries.events[0], self.entries.date)
        return

    def polan(self, se: ScaledEvent, date: dt.datetime):
        logger.info(f"Running POLAN for {date} on {se.description}")
        sd = SimulationDataset()
        h, fh, freqs, h_virtual = self.solving_integral(se)
        self.draw_traces(se, date, h, fh, h_virtual, freqs)
        return

    def draw_traces(
        self,
        se: ScaledEvent,
        date: dt.datetime,
        h: np.array,
        fh: np.array,
        h_virtual: np.array,
        freqs: np.array,
    ):
        dp = DigiPlots(
            fig_title="",
            nrows=1,
            ncols=1,
            font_size=10,
            figsize=(3, 3),
            date=date,
            date_lims=[],
            subplot_kw=None,
            draw_local_time=False,
        )
        ax = dp.get_axes(False)
        se.draw_trace(ax)
        ax.plot(fh, h, ls="-", lw=0.5, color="k")
        ax.plot(freqs, h_virtual, "g.", ms=0.7, alpha=0.8)
        ax.set_xlim(2, 8)
        ax.set_ylim(80, 400)
        dp.save(f"tmp/polan/{date.strftime('%Y%m%d%H%M')}.png")
        dp.close()
        return

    def solving_integral(
        self,
        se: ScaledEvent,
        h_base: float = 70,
        model_ionosphere: Union[str] = "parabolic",
    ):
        freqs = np.linspace(se.fv.min(), se.fv.max(), 101)
        omega = 2 * np.pi * freqs
        wave_number = omega / pconst["c"]
        if model_ionosphere == "parabolic":
            (h, fh) = self.parabolic_ionosphere(
                self.nbins,
                self.h_steps,
                ["E", "F"],
                [50, 80],
                [3.9, se.fv.max()],
                [125, 260],
            )
        h_virtual = np.zeros_like(freqs)
        base_index = h.tolist().index(h_base)
        for i, f in enumerate(freqs):
            X = (fh / f) ** 2
            m_index = np.where(X >= 1)[0][0]
            u = 1 / (np.sqrt(1 - X[base_index:m_index]))
            h_reflection = h[m_index]
            h_virtual[i] = h_base + np.trapz(u, x=None, dx=self.h_steps)
        return h, fh, freqs, h_virtual

    def parabolic_ionosphere(
        self,
        nbins: int,
        h_step: float,
        regions: List[str],
        ds: List[float],
        fps: List[float],
        hps: List[float],
    ):
        h = np.arange(self.nbins) * h_step
        fh = np.zeros(self.nbins)
        for region, d, fp, hp in zip(regions, ds, fps, hps):
            f = np.zeros(self.nbins)
            mask = np.abs(h - hp) < d
            f[mask] = (fp * (1 - ((h - hp) / d) ** 2))[mask]
            fh += f
        return (h, fh)


if __name__ == "__main__":
    file_path = "tmp/polan/ionogram_data.json"
    e = ScaledEntries.load_file(file_path)
    p = Polan(e)
