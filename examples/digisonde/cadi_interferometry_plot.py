"""CADI ionogram and interferometry workflow.

This example demonstrates three CADI product levels:

1) ``product="products"``
   Power, phase, Doppler-bin, baseline phase, and coherence. This is the
   safest product for routine power/Doppler ionogram plots.

2) ``product="interferometry"``
   Baseline-level phase/projection products. This works with two or more
   receivers, but two receivers only provide one-dimensional interferometry.

3) ``product="echoes"``
   Echo-level O/X mode classification. Angle-of-arrival products
   (``XL``, ``YL``, ``ZL`` and ground projection) are retained in the
   DataFrame for development. This example can plot them as diagnostic-only
   figures to show collaborators why CADI AOA calibration is still needed.

Before running on a new campaign, update ``folders`` and the station metadata.
For the SHA example, ``location.ini`` reports active receiver bitmask 14,
pulse rate 20 pps, and height range 1020 km. It does not provide the Doppler
FFT size, so this example leaves ``fft_size = None`` and plots Doppler bin.
Calibrated Doppler Hz is not plotted until the FFT size/bin convention is
confirmed. The transmitter and sampling delays are used to apply a small
virtual-height correction.

The standard Northern Hemisphere CADI layout is Rx1 north, Rx2 east, Rx3
south, and Rx4 west, with dipoles oriented 0/90/0/90 degrees. If a site's
receiver numbering or antenna orientation differs from that convention, set
``receiver_layout = CadiReceiverLayout.CUSTOM`` and pass explicit
``unit_vectors`` / ``dipole_orient_deg`` values to ``load_CADI_files``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pynasonde.digisonde import DigiPlots
from pynasonde.digisonde.cadi import CadiExtractor, CadiReceiverLayout


folders = ["tmp/CADI"]
lat = 17.47
lon = 78.57
diagonal_m = 30.0
orientation_deg = 0.0
rx_bitmask = 14  # File has 3 samples, physically mapped to Rx2 + Rx3 + Rx4.
pulse_rate_hz = 20.0
fft_size = None  # Set only if known; required for calibrated Doppler Hz.
height_range_km = 1020
transmitting_delay_us = 2056
sampling_delay_us = 2063
receiver_layout = CadiReceiverLayout.STANDARD

# For a pure rotation of the standard layout, change ``orientation_deg``.
plot_doppler_hz = False  # Pending confirmed FFT size and Doppler bin convention.
plot_aoa_products = True  # Diagnostic only; pending calibrated CADI AOA solution.

out_power = Path("docs/examples/figures/cadi_power_scatter.png")
out_doppler = Path("docs/examples/figures/cadi_doppler_scatter.png")
out_doppler_hz = Path("docs/examples/figures/cadi_doppler_hz_scatter.png")
out_modes = Path("docs/examples/figures/cadi_ox_mode_scatter.png")
out_xl_yl = Path("docs/examples/figures/cadi_xl_yl_diagnostic.png")
out_ground_projection = Path(
    "docs/examples/figures/cadi_ground_projection_diagnostic.png"
)


df_products = CadiExtractor.load_CADI_files(
    folders=folders,
    product="products",
    n_procs=1,
    transmitting_delay_us=transmitting_delay_us,
    sampling_delay_us=sampling_delay_us,
)

if df_products.empty:
    print("No CADI detections found. Check folders/exts.")
else:
    out_power.parent.mkdir(parents=True, exist_ok=True)

    power_plot = DigiPlots(
        figsize=(8, 4),
        fig_title="CADI Power Ionogram",
        draw_local_time=False,
    )
    power_plot.add_frequency_height_scatter(
        df_products,
        zparam="mean_power_db",
        prange=[0.0, 35.0],
        cbar_label="Mean Power, dB",
    )
    power_plot.save(str(out_power))
    power_plot.close()

    dop_plot = DigiPlots(
        figsize=(8, 4),
        fig_title="CADI Doppler Ionogram",
        draw_local_time=False,
    )
    dop_plot.add_frequency_height_scatter(
        df_products,
        zparam="doppler_bin",
        cmap="turbo",
        prange=[0.0, 7.0],
        cbar_label="Doppler Bin",
    )
    dop_plot.save(str(out_doppler))
    dop_plot.close()

    df_interferometry = CadiExtractor.load_CADI_files(
        folders=folders,
        product="interferometry",
        n_procs=1,
        lat=lat,
        lon=lon,
        diagonal_m=diagonal_m,
        orientation_deg=orientation_deg,
        rx_bitmask=rx_bitmask,
        receiver_layout=receiver_layout,
        transmitting_delay_us=transmitting_delay_us,
        sampling_delay_us=sampling_delay_us,
    )
    df_echoes = CadiExtractor.load_CADI_files(
        folders=folders,
        product="echoes",
        n_procs=1,
        lat=lat,
        lon=lon,
        diagonal_m=diagonal_m,
        orientation_deg=orientation_deg,
        rx_bitmask=rx_bitmask,
        receiver_layout=receiver_layout,
        fft_size=fft_size,
        pulse_rate_hz=pulse_rate_hz,
        transmitting_delay_us=transmitting_delay_us,
        sampling_delay_us=sampling_delay_us,
    )

    if not df_echoes.empty:
        if plot_doppler_hz and fft_size is not None:
            dop_hz_plot = DigiPlots(
                figsize=(8, 4),
                fig_title="CADI Doppler Frequency Ionogram",
                draw_local_time=False,
            )
            dop_hz_plot.add_frequency_height_scatter(
                df_echoes,
                zparam="doppler_hz",
                cmap="turbo",
                prange=[-pulse_rate_hz / 2.0, pulse_rate_hz / 2.0],
                cbar_label="Doppler Frequency, Hz",
            )
            dop_hz_plot.save(str(out_doppler_hz))
            dop_hz_plot.close()

        mode_plot = DigiPlots(
            figsize=(8, 4),
            fig_title="CADI O/X Mode Ionogram",
            draw_local_time=False,
        )
        mode_plot.add_categorical_frequency_height_scatter(
            df_echoes,
            category_param="mode",
            colors={
                "O": "#0C5DA5",
                "X": "#FF9500",
                "ambiguous": "#9E9E9E",
                "unknown": "#474747",
            },
            category_order=["O", "X", "ambiguous", "unknown"],
        )
        mode_plot.save(str(out_modes))
        mode_plot.close()

        if plot_aoa_products:
            mode_colors = {
                "O": "#0C5DA5",
                "X": "#FF9500",
                "ambiguous": "#9E9E9E",
                "unknown": "#474747",
            }

            fig, ax = plt.subplots(figsize=(5, 5))
            work = df_echoes.dropna(subset=["XL", "YL", "mode"])
            for mode, color in mode_colors.items():
                mode_df = work[work["mode"] == mode]
                if mode_df.empty:
                    continue
                ax.scatter(
                    mode_df["XL"],
                    mode_df["YL"],
                    s=5,
                    c=color,
                    alpha=0.75,
                    label=mode,
                )
            ax.set_xlabel("XL")
            ax.set_ylabel("YL")
            ax.set_title("CADI XL/YL Diagnostic\nAOA calibration pending")
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(out_xl_yl, dpi=300, bbox_inches="tight")
            plt.close(fig)

            valid_projection = np.isfinite(df_echoes["ZL"]) & (
                np.abs(df_echoes["ZL"]) > 1e-6
            )
            df_projection = df_echoes.loc[valid_projection].copy()
            df_projection["east_km"] = (
                df_projection["height_km"] * df_projection["XL"] / df_projection["ZL"]
            )
            df_projection["north_km"] = (
                df_projection["height_km"] * df_projection["YL"] / df_projection["ZL"]
            )

            fig, ax = plt.subplots(figsize=(5, 5))
            work = df_projection.dropna(subset=["east_km", "north_km", "mode"])
            for mode, color in mode_colors.items():
                mode_df = work[work["mode"] == mode]
                if mode_df.empty:
                    continue
                ax.scatter(
                    mode_df["east_km"],
                    mode_df["north_km"],
                    s=5,
                    c=color,
                    alpha=0.75,
                    label=mode,
                )
            ax.set_xlim(-200, 200)
            ax.set_ylim(-200, 200)
            ax.set_xlabel("East Projection, km")
            ax.set_ylabel("North Projection, km")
            ax.set_title("CADI Ground Projection Diagnostic\nAOA calibration pending")
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(out_ground_projection, dpi=300, bbox_inches="tight")
            plt.close(fig)

    print("Saved:", out_power)
    print("Saved:", out_doppler)
    if not df_echoes.empty:
        if plot_doppler_hz and fft_size is not None:
            print("Saved:", out_doppler_hz)
        else:
            print(
                "Skipped Doppler Hz plot: FFT size/bin convention pending "
                "calibration. Use Doppler bin for now."
            )
        print("Saved:", out_modes)
        if plot_aoa_products:
            print(
                "Saved diagnostic AOA plots: calibration pending; do not use "
                "XL/YL/ZL or ground projection as validated science products."
            )
            print("Saved:", out_xl_yl)
            print("Saved:", out_ground_projection)
        else:
            print(
                "Skipped XL/YL/ZL and ground-projection plots: CADI AOA "
                "calibration is pending. O/X mode classification is retained."
            )
    print("Product rows:", len(df_products))
    print("Interferometry rows:", len(df_interferometry))
    print("Echo rows:", len(df_echoes))
    if not df_echoes.empty and "height_correction_km" in df_echoes:
        print(
            "Height correction km:",
            float(df_echoes["height_correction_km"].dropna().iloc[0]),
        )
    if not df_interferometry.empty:
        print(
            "Interferometry columns:",
            [
                "rx_a",
                "rx_b",
                "phase_deg",
                "coherence",
                "projected_direction_cosine",
            ],
        )
    if not df_echoes.empty:
        if plot_doppler_hz and fft_size is not None:
            print(
                "Doppler Hz range:",
                (
                    float(df_echoes["doppler_hz"].min()),
                    float(df_echoes["doppler_hz"].max()),
                ),
            )
        else:
            print(
                "Doppler Hz range unavailable: provide fft_size for calibration."
            )
        print("AOA method counts:", df_echoes["aoa_method"].value_counts().to_dict())
        print("Echo mode counts:", df_echoes["mode"].value_counts().to_dict())
