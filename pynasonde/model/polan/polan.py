import datetime as dt
from typing import List, Union

import numpy as np
from loguru import logger

from pynasonde.digisonde.digi_plots import DigiPlots
from pynasonde.model.absorption.constants import pconst
from pynasonde.model.polan.datasets import ScaledEntries, ScaledEvent, SimulationOutputs
from pynasonde.model.polan.polan_utils import (
    chapman_ionosphere,
    ne2f,
    parabolic_ionosphere,
)


class Polan(object):
    """
    ** This is an python implementation of method described in
    https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2011RS004833

    A generalised method to .


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
        fig_file_name: str = None,
    ):
        self.entries = entries
        self.h_max_simulation = h_max_simulation
        self.h_steps = h_steps
        if type(h_steps) == float:
            self.nbins = int(h_max_simulation / h_steps)
        self.fig_file_name = fig_file_name
        return

    def polan(
        self,
        se: ScaledEvent,
        date: dt.datetime,
        h_base: float = 70,
        model_ionospheres: List[dict] = [
            dict(
                model="Chapman",
                layer="F",
                Np=6.1e11,
                hp=250,
                scale_h=45,
            ),
        ],
        index: int = 0,
        plot: bool = False,
    ):
        se = se if se else self.entries.events[index]
        logger.info(f"Running POLAN for {date} on {se.description}")
        sd = self.solving_integral(se, h_base, model_ionospheres)

        if plot:
            self.draw_traces(se, date, h, fh, h_virtual, freqs)
        return sd

    def solving_integral(
        self,
        se: ScaledEvent,
        h_base: float = 70,
        model_ionospheres: List[dict] = [
            dict(
                model="Chapman",
                layer="F",
                Np=6.1e11,
                hp=250,
                scale_h=45,
            ),
        ],
    ):
        freqs = np.linspace(se.fv.min(), se.fv.max(), 101)
        fhs, hs = None, None
        for model_ionosphere in model_ionospheres:
            if "Parabolic" == model_ionosphere["model"]:
                (h, fh) = parabolic_ionosphere(
                    self.nbins,
                    self.h_steps,
                    [model_ionosphere["layer"]],
                    [model_ionosphere["D"]],
                    [model_ionosphere["Np"]],
                    [model_ionosphere["hp"]],
                )
            if "Chapman" == model_ionosphere["model"]:
                (h, fh) = chapman_ionosphere(
                    self.nbins,
                    self.h_steps,
                    [model_ionosphere["layer"]],
                    [model_ionosphere["Np"]],
                    [model_ionosphere["hp"]],
                    [model_ionosphere["scale_h"]],
                )
            if hs is None:
                hs, fhs = h, fh
            else:
                fhs += fh
        hvs = np.zeros_like(freqs) * np.nan
        base_index = hs.tolist().index(h_base)
        for i, f in enumerate(freqs):
            X = (fhs / f) ** 2
            if np.nanmax(X) >= 1.0:
                m_index = np.where(X >= 1)[0][0]
                u = 1 / (np.sqrt(1 - X[base_index:m_index]))
                # h_reflection = h[m_index]
                hvs[i] = h_base + np.trapz(u, x=None, dx=self.h_steps)
        sd = SimulationOutputs(h=hs, fh=fhs, tf_sweeps=freqs, h_virtual=hvs)
        return sd

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
        if self.fig_file_name:
            dp.save(self.fig_file_name)
        dp.close()
        return


if __name__ == "__main__":
    file_path = "tmp/polan/ionogram_data.json"
    e = ScaledEntries.load_file(file_path)
    p = Polan(e)
