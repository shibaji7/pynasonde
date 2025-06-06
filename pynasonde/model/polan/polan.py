import copy
import datetime as dt
from typing import List, Union

import numpy as np
from joblib import Parallel, delayed
from loguru import logger

from pynasonde.digisonde.digi_plots import DigiPlots
from pynasonde.model.absorption.constants import pconst
from pynasonde.model.polan.datasets import SimulationOutputs, Trace
from pynasonde.model.polan.polan_utils import (
    chapman_ionosphere,
    generate_random_samples,
    get_Np_bounds_from_fv,
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
        trace: Trace,
        h_max_simulation: float = 500,
        h_steps: Union[float, str] = 1e-3,
        fig_file_name: str = None,
        optimize: bool = False,
    ):
        self.trace = trace
        self.h_max_simulation = h_max_simulation
        self.h_steps = h_steps
        if type(h_steps) == float:
            self.nbins = int(h_max_simulation / h_steps)
        self.fig_file_name = fig_file_name
        self.optimize = optimize
        return

    def polan(
        self,
        date: dt.datetime,
        trace: Trace = None,
        h_base: float = 70,
        optimzer_n_samples: int = 100,
        model_ionospheres: List[dict] = [
            dict(
                model="Chapman",
                layer="F",
                Np=6.1e11,
                hp=250,
                scale_h=45,
            ),
        ],
        plot: bool = False,
        run_Es_only: bool = False,
        n_jobs: int = 8,
    ):
        trace = trace if trace else self.trace
        logger.info(f"Running POLAN for {date} on {self.trace.filename}")
        if self.optimize:
            sd = self.run_optimizer(
                trace,
                h_base,
                model_ionospheres,
                optimzer_n_samples,
                run_Es_only,
                n_jobs,
            )
        else:
            sd = self.run_solver(trace, h_base, run_Es_only, model_ionospheres)
        if plot:
            self.draw_traces(self.trace, date, sd)
        return sd

    def run_solver(
        self,
        trace: Trace,
        h_base: float = 70,
        run_Es_only: bool = False,
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
        if run_Es_only and "Es" in [mi["layer"] for mi in model_ionospheres]:
            trace_Es, trace_no_Es = copy.copy(trace), copy.copy(trace)
            trace_Es.events, trace_no_Es.events = (
                [te for te in trace_Es.events if "Es" in te.description],
                [te for te in trace_no_Es.events if "Es" not in te.description],
            )
            sd = [
                self.__integral__(
                    trace_Es,
                    h_base,
                    [mi for mi in model_ionospheres if mi.get("layer") == "Es"],
                ),
                self.__integral__(
                    trace_no_Es,
                    h_base,
                    [mi for mi in model_ionospheres if mi.get("layer") != "Es"],
                ),
            ]
        else:
            sd = [self.__integral__(trace, h_base, model_ionospheres)]
        return sd

    def run_optimizer(
        self,
        trace: Trace,
        h_base: float = 70,
        model_ionospheres: List[dict] = [
            dict(
                model="Chapman",
                layer="F",
                np_bounds=[1e11, 6.1e11],
                hp_bounds=[250, 300],
                hd_bounds=[45, 70],
            ),
        ],
        optimzer_n_samples: int = 100,
        run_Es_only: bool = False,
        n_jobs: int = 8,
    ):
        logger.info(f"Running optimizer....")
        [
            mi.update(
                dict(
                    samples=generate_random_samples(
                        mi["hp_bound"],
                        mi["np_bound"],
                        mi["hd_bound"],
                        optimzer_n_samples,
                    )
                )
            )
            for mi in model_ionospheres
        ]
        self.sol_ionospheres, self.hv_errs = [], []
        self.sol_ionospheres = Parallel(n_jobs=n_jobs)(
            delayed(self.run_solver)(
                trace,
                h_base,
                run_Es_only,
                [
                    dict(
                        model=mi["model"],
                        layer=mi["layer"],
                        hp=mi["samples"][j, 0],
                        Np=mi["samples"][j, 1],
                        scale_h=mi["samples"][j, 2],
                    )
                    for mi in model_ionospheres
                ],
            )
            for j in range(optimzer_n_samples)
        )
        self.hv_errs = [
            np.nansum([s.hv_err for s in sd]) for sd in self.sol_ionospheres
        ]
        sd_min = self.sol_ionospheres[np.nanargmin(self.hv_errs)]
        return sd_min

    def __integral__(
        self,
        se: Trace,
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
        fmin, fmax = (
            np.min([e.fv.min() for e in se.events]),
            np.max([e.fv.max() for e in se.events]),
        )
        tfreq = np.array([x for e in se.events for x in e.fv])
        freqs = np.linspace(fmin, fmax, 101)
        fhs, hs = None, None
        for model_ionosphere in model_ionospheres:
            if "Parabolic" == model_ionosphere["model"]:
                (h, fh) = parabolic_ionosphere(
                    self.nbins,
                    self.h_steps,
                    [model_ionosphere["layer"]],
                    [model_ionosphere["scale_h"]],
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
        hvs = self.compute_O_mode_ref(freqs, hs, fhs, h_base)
        hvs_e = self.compute_O_mode_ref(tfreq, hs, fhs, h_base)
        sd = SimulationOutputs(
            h=hs,
            fh=fhs,
            tf_sweeps=freqs,
            h_virtual=hvs,
            tf_sweeps_e=tfreq,
            h_virtual_e_model=hvs_e,
            h_virtual_e_obs=np.array([x for e in se.events for x in e.ht]),
        )
        logger.info(f"RMdSE: {sd.compute_rMdse()}")
        return sd

    def compute_O_mode_ref(self, freqs, hs, fhs, h_base):
        hvs = np.zeros_like(freqs) * np.nan
        base_index = hs.tolist().index(h_base)
        for i, f in enumerate(freqs):
            X = (fhs / f) ** 2
            if np.nanmax(X) >= 1.0:
                m_index = np.where(X >= 1)[0][0]
                u = 1 / (np.sqrt(1 - X[base_index:m_index]))
                hvs[i] = h_base + np.trapz(u, x=None, dx=self.h_steps)
        return hvs

    def draw_traces(
        self,
        se: Trace,
        date: dt.datetime,
        sd_list: List[SimulationOutputs],
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
        for sd in sd_list:
            ax.plot(sd.fh, sd.h, ls="-", lw=0.5, color="k")
            ax.scatter(
                sd.tf_sweeps, sd.h_virtual, color="g", marker="s", s=0.5, alpha=0.8
            )
        ax.set_xlim(1, 10)
        ax.set_ylim(80, 600)
        if self.fig_file_name:
            dp.save(self.fig_file_name)
        dp.close()
        return


if __name__ == "__main__":
    file_path = "tmp/20250527/KW009_2025147120000_SAO.XML"
    e = Trace.load_xml_sao_file(file_path)[0]
    p = Polan(
        e, fig_file_name="tmp/polan/sample.png", h_max_simulation=700, optimize=True
    )
    p.polan(
        dt.datetime(2025, 5, 27),
        model_ionospheres=[
            dict(
                model="Chapman",
                layer="F",
                np_bound=get_Np_bounds_from_fv(
                    [x for ex in e.events for x in ex.fv], df=0.3
                ),
                hp_bound=[300, 400],
                hd_bound=[30, 60],
            ),
            dict(
                model="Parabolic",
                layer="Es",
                np_bound=get_Np_bounds_from_fv(
                    [x for ex in e.events for x in ex.fv if "Es" in ex.description],
                    df=0.3,
                ),
                hp_bound=[100, 110],
                hd_bound=[3, 8],
            ),
        ],
        plot=True,
        run_Es_only=True,
        n_jobs=24,
    )

    # p = Polan(e, fig_file_name="tmp/polan/sample.png", h_max_simulation=700)
    # p.polan(
    #     dt.datetime(2025, 5, 27),
    #     model_ionospheres=[
    #         dict(
    #             model="Chapman",
    #             layer="F",
    #             Np=[f2ne(8.1)],
    #             hp=[400],
    #             scale_h=[60],
    #         ),
    #         dict(
    #             model="Parabolic",
    #             layer="Es",
    #             Np=[f2ne(2.5)],
    #             hp=[105],
    #             scale_h=[5],
    #         ),
    #     ],
    #     plot=True,
    #     run_Es_only=True,
    # )
