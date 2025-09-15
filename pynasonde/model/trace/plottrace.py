#!/usr/bin/env python

"""rtplots.py: Calculate all the functions of utility plots"""

__author__ = "Chakraborty, S."
__copyright__ = "Chakraborty, S."
__credits__ = []
__license__ = "MIT"
__version__ = "1.0."
__maintainer__ = "Chakraborty, S."
__email__ = "chakras4@erau.edu"
__status__ = "Research"

import matplotlib.pyplot as plt

# import scienceplots
import scienceplots

plt.style.use(["science", "ieee"])
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Tahoma", "DejaVu Sans", "Lucida Grande", "Verdana"]
plt.rcParams["text.usetex"] = False
import numpy as np


class PlotRays(object):
    def __init__(
        self,
        nrows=1,
        ncols=1,
        ylim=[],
        xlim=[],
        oth=True,
        figsize=(5, 5),
        Re_km=6371.0,
    ):
        self.nrows = nrows
        self.ncols = ncols
        self.xlim = xlim
        self.ylim = ylim
        self.axnum = 0
        self.fig = plt.figure(figsize=(figsize[0] * ncols, figsize[1] * nrows), dpi=300)
        self.oth = oth
        self.Re = Re_km
        return

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight", facecolor=(1, 1, 1, 1))
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return

    def get_parameter(self, kind):
        import matplotlib.colors as colors

        if kind == "pf":
            o, cmap, label, norm = (
                getattr(self, kind),
                "PuOr",
                # "YlGnBu",
                r"$f_0$ [MHz]",
                colors.Normalize(1, 9),
            )
        if kind == "edens":
            o, cmap, label, norm = (
                getattr(self, kind),
                "cool",
                r"$N_e$ [$m^{-3}$]",
                colors.LogNorm(1e10, 1e12),
            )
        if kind == "ref_indx":
            o, cmap, label, norm = (
                getattr(self, kind),
                "cool",
                r"$\eta$",
                colors.Normalize(0.8, 1),
            )
        return o, cmap, label, norm

    def get_arc_heights(self, height, dist):
        darc = dist / self.Re
        true_height = self.Re + height
        height = true_height * np.cos(darc) - self.Re
        return height

    def create_figure_pane(self, xlabel=r"Ground range, km", ylabel=r"Height, km"):
        self.axnum += 1
        fignum = 100 * self.nrows + 10 * self.ncols + self.axnum
        ax = self.fig.add_subplot(fignum)
        # Create Arc
        if self.oth:
            theta = np.deg2rad(np.linspace(-180, 180, 181))
            x, y = self.Re * np.cos(theta), self.Re * np.sin(theta) - self.Re
            ax.plot(x, y, ls="-", color="k", lw=1)
            ax.text(
                -400,
                200,
                ylabel,
                ha="left",
                va="center",
                fontdict={"size": 12, "fontweight": "bold"},
                rotation=90,
            )
            ax.text(
                0,
                -50,
                xlabel,
                ha="center",
                va="top",
                fontdict={"size": 12, "fontweight": "bold"},
            )
            ax.set_facecolor("0.98")
            ax.fill_between(x, -800 * np.ones_like(y), y, color="gray", alpha=0.5)
        else:
            ax.set_ylabel(ylabel, fontdict={"size": 12, "fontweight": "bold"})
            ax.set_xlabel(xlabel, fontdict={"size": 12, "fontweight": "bold"})
        ax.set_xlim(self.xlim if len(self.xlim) == 2 else [-300, 300])
        ax.set_ylim(self.ylim if len(self.ylim) == 2 else [-100, 800])
        ax.tick_params(axis="both", labelsize=11)
        ax.set_yticks([0, 200, 400, 600, 800])
        return ax

    def lay_rays(
        self,
        outputs=[],
        kind="edens",
        lcolor="k",
        lw=0.3,
        ls="-",
        param_alpha=1,
        tag_distance: float = -1,
        ax=None,
        xlabel=r"Ground range, km",
        ylabel=r"Height, km",
        date=None,
        stitle=None,
        text="(A)",
        ped_angles=[],
        add_cbar=True,
        param_zorder=2,
        ray_zorder=3,
    ):
        ax = ax if ax else self.create_figure_pane(xlabel, ylabel)
        o, cmap, label, norm = self.get_parameter(kind)

        im = ax.pcolormesh(
            self.X,
            self.Z,
            o,
            norm=norm,
            cmap=cmap,
            alpha=param_alpha,
            zorder=param_zorder,
        )
        if add_cbar:
            pos = ax.get_position()
            cpos = [
                pos.x1 + 0.025,
                pos.y0 + 0.05,
                0.015,
                pos.height * 0.6,
            ]
            cax = self.fig.add_axes(cpos)
            cbax = self.fig.colorbar(
                im, cax, spacing="uniform", orientation="vertical", cmap="plasma"
            )
            _ = cbax.set_label(label, fontsize=11)
            cbax.ax.tick_params(axis="both", labelsize=11)

        for o in outputs:
            x_km, y_km = o.x_km, o.y_km
            if self.oth:
                y_km = self.get_arc_heights(y_km, x_km)
            col, width = lcolor, lw
            if o.el0_deg in ped_angles:
                col, width = "darkgreen", lw * 2
            ax.plot(x_km, y_km, c=col, zorder=ray_zorder, ls=ls, lw=width)
        if text:
            ax.text(0.05, 0.95, text, ha="left", va="top", transform=ax.transAxes)
        return ax

    def set_density(self, X, Z, Ne, pf=None):
        self.X, self.Z, self.edens = X, Z, Ne
        self.pf = pf
        if self.oth:
            self.Z = self.get_arc_heights(self.Z, self.X)
        return
