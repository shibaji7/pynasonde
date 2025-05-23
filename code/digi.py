import os
from datetime import datetime

from pynasonde.digisonde.digi_plots import SaoSummaryPlots
from pynasonde.digisonde.raw.iqstream_afrl import IQStreamReader
from pynasonde.digisonde.raw.parse import IQDigisonde, Program

if __name__ == "__main__":

    d = datetime(2023, 10, 14, 15, 56)
    r = IQStreamReader(d)

    p = Program(
        epoch=d,
        fft_mode=False,
        rx_tag="ch1",
        save_phase=True,
        signal_type="DPS4D",
        id="DPS4D_Kirtland0",
        freq_sampling_law="linear",
        lower_freq_limit=2e6,
        upper_freq_limit=15e6,
        coarse_freq_step=30e3,
        number_of_fine_steps=1,
        fine_frequency_step=5e3,
        fine_muliplexing=False,
        inter_pulse_period=2 * 5e-3,
        number_of_integrated_pulses=8,
        inter_pulse_phase_switch=False,
        waveform_type="16-chip complementary",
        polarization="O and X",
        out_dir="./tmp/out/",
    )

    dg = IQDigisonde(p, min_range=0, max_range=1000)
    ds = dg.process(r.read_file())
    s = SaoSummaryPlots(
        "",
        date=datetime(2023, 10, 14, 15, 56),
    )
    ax = s.get_axes(False)
    ax.set_xlim(1.8, 22)
    ax.set_ylim(50, 600)
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_ylabel("Virtual Height", fontsize=12)
    ax.set_title("DPS4D Kirtland0 / 14 Oct 2023 / PSD", fontsize=12)
    print(ds.variables["power_O"])
    print(ds.variables["frequency"])
    print(ds.variables["range"])
    ax.pcolormesh(
        ds.variables["frequency"] / 1e6,
        ds.variables["range"],
        ds.variables["power_O"].T,
        shading="auto",
        cmap="jet",
    )
    s.save(
        os.path.join(
            p.out_dir,
            os.uname().nodename,
            p.id,
            p.epoch.strftime("%Y-%m-%d"),
            f"{p.id}_{p.epoch.strftime('%Y-%m-%d_%H%M%S')}.png",
        )
    )
    s.close()
