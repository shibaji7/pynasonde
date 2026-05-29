"""MkDocs example for multi-record (day-file) SAO workflows.

This script demonstrates three practical paths for day-style `.SAO` files:

1) Full-day height-profile extraction (`mode="auto"`):
   - Load all records from one or more folders.
   - Plot electron density as a time-height panel.

2) Full-day scaled-parameter extraction (`mode="auto"`):
   - Reload using `func_name="scaled"`.
   - Plot `foF2` and `hmF2` on dual y-axes.

3) Indexed record extraction (`mode="single"` + `record_index`):
   - Pull one selected scan from a day file.
   - Pull the last scan using a negative index.
   - Compare those scan-level points in a compact panel.

Update `folders`, `date`, and plot limits to match your campaign.
"""

import datetime as dt
from pathlib import Path

import matplotlib.dates as mdates
import pandas as pd

from pynasonde.digisonde.digi_plots import SaoSummaryPlots
from pynasonde.digisonde.digi_utils import setsize
from pynasonde.digisonde.parsers.sao import SaoExtractor

# ----------------------------
# User-facing configuration
# ----------------------------
date = dt.datetime(2024, 10, 25)
folders = ["tmp/data/"]
n_procs = 12
font_size = 16
day_mode = "auto"
selected_record_index = 10
height_prange = [0, 3]

fig_dir = Path("docs/examples/figures")
tmp_dir = Path("tmp/pynasondev1")
fig_dir.mkdir(parents=True, exist_ok=True)
tmp_dir.mkdir(parents=True, exist_ok=True)

setsize(font_size)


def _set_day_axis(ax, base_date):
    """Constrain x-axis to one day with 6-hour ticks."""
    ax.set_xlim([base_date, base_date + dt.timedelta(1)])
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))


def _dedupe_legend(plot_obj):
    """Combine and de-duplicate legends across all figure axes."""
    main_ax = plot_obj.axes
    handles, labels = [], []
    for axis in plot_obj.fig.get_axes():
        h, l = axis.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    seen = set()
    u_handles, u_labels = [], []
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen.add(l)
            u_handles.append(h)
            u_labels.append(l)
    if u_handles:
        main_ax.legend(u_handles, u_labels, loc=2)


# ---------------------------------
# 1) Full-day height-profile panel
# ---------------------------------
df_hp = SaoExtractor.load_SAO_files(
    folders=folders,
    func_name="height_profile",
    n_procs=n_procs,
    mode=day_mode,
)

if len(df_hp) == 0:
    raise RuntimeError(
        "No height-profile SAO rows were loaded. Check `folders` and file pattern."
    )

# Convert electron density to 10^6 cm^-3 for a compact color scale.
df_hp = df_hp.copy()
df_hp["ed"] = df_hp["ed"] / 1e6

sao_plot = SaoSummaryPlots(
    figsize=(8, 4),
    fig_title="JI91J / Ne Profiles (day-file multi-record), 25 Oct 2024",
    draw_local_time=False,
    font_size=font_size,
)
sao_plot.add_TS(
    df_hp,
    zparam="ed",
    prange=height_prange,
    zparam_lim=10,
    cbar_label=r"$N_e$,$\times 10^{6}$ /cc",
    plot_type="scatter",
    scatter_ms=20,
)
_set_day_axis(sao_plot.axes, date)
sao_plot.save(str(fig_dir / "stack_sao_multi_ne.png"))
sao_plot.fig.savefig(
    str(tmp_dir / "PynasondeV1-SAO-Multi-01.png"),
    format="png",
    dpi=300,
    bbox_inches="tight",
)
sao_plot.close()


# -------------------------------------
# 2) Full-day scaled-parameter panel
# -------------------------------------
df_scaled = SaoExtractor.load_SAO_files(
    folders=folders,
    func_name="scaled",
    n_procs=n_procs,
    mode=day_mode,
)

if len(df_scaled) == 0:
    raise RuntimeError(
        "No scaled SAO rows were loaded. Check `folders` and file pattern."
    )

sao_plot = SaoSummaryPlots(
    figsize=(8, 4),
    fig_title="JI91J / F2 (scaled, day-file multi-record), 25 Oct 2024",
    draw_local_time=False,
    font_size=font_size,
)
sao_plot.plot_TS(
    df_scaled,
    right_yparams=["hmF2"],
    left_yparams=["foF2"],
    right_ylim=[100, 450],
    left_ylim=[1, 15],
)
_set_day_axis(sao_plot.axes, date)
_dedupe_legend(sao_plot)
sao_plot.save(str(fig_dir / "stack_sao_multi_F2.png"))
sao_plot.fig.savefig(
    str(tmp_dir / "PynasondeV1-SAO-Multi-02.png"),
    format="png",
    dpi=300,
    bbox_inches="tight",
)
sao_plot.close()


# -------------------------------------------------------
# 3) Advanced indexed-scan extraction and comparison plot
# -------------------------------------------------------
df_selected = SaoExtractor.load_SAO_files(
    folders=folders,
    func_name="scaled",
    n_procs=n_procs,
    mode="single",
    record_index=selected_record_index,
)
df_last = SaoExtractor.load_SAO_files(
    folders=folders,
    func_name="scaled",
    n_procs=n_procs,
    mode="single",
    record_index=-1,
)

df_selected = df_selected.copy()
df_last = df_last.copy()
df_selected["series"] = f"record_index={selected_record_index}"
df_last["series"] = "record_index=-1 (last)"
df_compare = pd.concat([df_selected, df_last], ignore_index=True)

sao_plot = SaoSummaryPlots(
    figsize=(8, 4),
    fig_title="JI91J / Indexed scans (single-record mode), 25 Oct 2024",
    draw_local_time=False,
    font_size=font_size,
)
sao_plot.plot_TS(
    df_compare,
    right_yparams=["hmF2"],
    left_yparams=["foF2"],
    right_ylim=[100, 450],
    left_ylim=[1, 15],
)
_set_day_axis(sao_plot.axes, date)
_dedupe_legend(sao_plot)
sao_plot.save(str(fig_dir / "stack_sao_multi_indexed.png"))
sao_plot.fig.savefig(
    str(tmp_dir / "PynasondeV1-SAO-Multi-03.png"),
    format="png",
    dpi=300,
    bbox_inches="tight",
)
sao_plot.close()
