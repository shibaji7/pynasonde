"""Functions to help read in raw binary data created by SDL/AFRL HF receivers.
Based on code written by Eugene Dao (AFRL). 
@author Riley Troyer, Space Dynamics Laboratory, riley.troyer@sdl.usu.edu

Copyright Space Dynamics Laboratory, 2025. The U.S. Federal Government retains a royalty-free, non-exclusive, non-
transferable license to read_binary_iq_sdl_afrl_receivers.py pursuant to
DFARS 252.227-7014. All other rights reserved.
"""

import os

# Needed libaries
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import pyfftw
import scipy.constants as C
from loguru import logger

from pynasonde.digisonde.raw.raw_plots import AFRLPlots


class IQStream(object):
    """Class to read in the binary data created by SDL/AFRL radar.
    INPUT
    t0 - when to start reading data, should be start time of file
    t_duration - how much data to read, should usually just be the file duration which is 1 second
                 datetime.timedelta(seconds=1)
    channel - what radar channel to read from
    data_dir - where raw binary files are stored
    OUTPUT
    samples - signal data
    """

    def __init__(
        self,
        t0: datetime,
        t_duration: timedelta = timedelta(seconds=1),
        channel: int = 0,
        data_dir: str = "/media/chakras4/69F9D939661D263B/",
        file_name_ts: str = "%Y-%m-%d_%H%M*",
    ):
        """
        This class holds the raw datasets.
        """
        self.t0 = t0
        self.t_duration = t_duration
        self.channel = channel
        self.data_dir = os.path.join(
            data_dir,
            str(self.t0.hour),
            str(self.t0.minute),
        )
        self.file_name_ts = file_name_ts
        logger.info(f"Data directory: {self.data_dir}")
        self.search_files()
        return

    def search_files(self):
        """
        Search for all .bin files under the data_dir
        """
        # Which receiver channel to use
        rx_tag = f"ch{self.channel}"
        # Filename
        data_filename = os.path.join(
            self.data_dir, self.t0.strftime(self.file_name_ts) + rx_tag + "*.bin"
        )
        files = glob(data_filename)
        files.sort()
        self.files = []
        for f in files:
            name_tags = f.split("/")[-1].split("_")[:2]
            t = datetime.strptime(f"{name_tags[0]}T{name_tags[1]}", "%Y-%m-%dT%H%M%S")
            if (t - self.t0) < self.t_duration:
                logger.info(f"Set file {f}")
                self.files.append(f)
        logger.info(f"Files to be read under {self.data_dir}: {len(self.files)}")
        return

    def read_file(self, f: str = None, index: int = 0):
        """
        Read files/sample under each .bin file
        """
        t0 = self.t0
        f = f if f else self.files[index]
        ds = dict()
        ds["start_time"] = t0
        # Get parameters from filename
        ds["i_khz_1"] = f.find("kHz")
        ds["i_khz_2"] = f.find("kHz", ds["i_khz_1"] + 1)
        ds["i_fc"] = f.find("fc")
        ds["i_bw"] = f.find("bw")

        # Get the center frequency in Hz
        ds["center_freq"] = float(f[ds["i_fc"] + 2 : ds["i_khz_1"]]) * 1e3

        # Get the band width frequency in Hz
        ds["sample_freq"] = float(f[ds["i_bw"] + 2 : ds["i_khz_2"]]) * 1e3

        # Max number of samples that can be read in
        ds["max_n_samples"] = int(ds["sample_freq"] * (1000000 - t0.microsecond) * 1e-6)
        # Calculate number of samples to read
        ds["n_samples"] = int(1 * ds["sample_freq"])

        # Create array to store samples in
        ds["samples"] = np.empty(ds["n_samples"], dtype="complex64")

        i_collected = 0
        n_to_collect = ds["n_samples"]
        # Open the data file
        with open(f, "rb") as file:

            # Loop through by specified number of samples
            while n_to_collect > 0:

                # Where to find data in file
                i_sample = int(round(t0.microsecond * 1e-6 * ds["sample_freq"]))

                # Start file at this location, move forward in each loop
                file.seek(4 * i_sample)

                # Calculate number of subsamples to read in, capped by samples left in file
                n_subsamples = min(n_to_collect, ds["max_n_samples"])

                # Read in data from file in int16 format
                subsamples = np.fromfile(file, dtype="int16", count=2 * n_subsamples)

                # Convert datatype from int16 to float 32 and to complex64
                subsamples = subsamples.astype("float32", copy=False)

                # Convert to complex. Every other value is the imaginary component
                subsamples = subsamples.view("complex64")

                # Change number of subsamples to reflect complex numbers
                n_subsamples = len(subsamples)

                # Write subsamples to main array
                ds["samples"][i_collected : (i_collected + n_subsamples)] = subsamples

                # Update variables and time
                i_collected = i_collected + n_subsamples
                n_to_collect = n_to_collect - n_subsamples
                t0 = t0 + timedelta(microseconds=n_subsamples / ds["sample_freq"] * 1e6)
        ds["end_time"] = t0
        ds["us"] = np.linspace(
            0, i_collected / ds["sample_freq"] * 1e6, int(ds["sample_freq"])
        )  # in microseconds
        ds["range"] = 0.5 * C.c * ds["us"] * 1e-6  # to seconds and range in meters
        logger.info(f"Sample data types: {ds['samples'].dtype}")
        return ds

    def to_pyfftw(
        self, ds: dict = None, f: str = None, index: int = 0, num_threads: int = 10
    ):
        """
        Convert to FFT and PSD with frequencies
        """
        ds = ds if ds else self.read_file(f, index)
        # Number of datapoints
        N = ds["n_samples"]
        # Initialze the fftw setups
        pyfftw.config.NUM_THREADS = num_threads
        frq_domain = pyfftw.empty_aligned(N, dtype="complex64")

        # Setup frequency components
        T = ds["us"][-1] * 1e-6  # Total time duration (seconds)
        fs = N / T  # Sampling frequency (Hz)
        f_fft = np.fft.fftfreq(N, d=1 / fs)  # Frequency vector (for NumPy's FFT)
        f_fft_positive = f_fft[: N // 2]  # positive frequency range

        # Create FFT object
        fft_obj = pyfftw.FFTW(
            ds["samples"],
            frq_domain,
            direction="FFTW_FORWARD",
            flags=("FFTW_ESTIMATE",),
        )
        fft_obj()
        psd = np.abs(frq_domain) ** 2 / (fs * N)
        ds["fft"] = dict(
            T=T,
            fs=fs,
            f_fft=f_fft,
            f_fft_positive=f_fft_positive,
            frq_domain=frq_domain,
            psd=psd,
        )
        return ds

    def to_spectrogram(
        self,
        ds: dict = None,
        f: str = None,
        index: int = 0,
        num_threads: int = 10,
        nfft: int = 1024,
        window: str = "hann",
        return_onesided: bool = False,
        mode: str = "complex",
        **kwrads,
    ) -> dict:
        """
        Convert to PSD with frequencies and time/range
        """
        ds = ds if ds else self.read_file(f, index)
        # Setup frequency components
        N = ds["n_samples"]
        T = ds["us"][-1] * 1e-6  # Total time duration (seconds)
        fs = N / T  # Sampling frequency (Hz)
        from scipy import signal

        f, t_spec, Sxx = signal.spectrogram(
            ds["samples"],
            fs=fs,
            nfft=nfft,
            window=window,
            return_onesided=return_onesided,
            mode=mode,
            **kwrads,
        )
        f = np.fft.fftshift(f)
        Sxx_db = np.fft.fftshift(10 * np.log10(np.abs(Sxx) ** 2 + 1e-12), axes=0)
        ds["spectrogram"] = dict(
            T=T,
            fs=fs,
            f=f,
            t_spec=t_spec,
            range_spec=t_spec * C.c / 2000,  # in km
            psd=Sxx_db,
        )
        logger.info(f"Shape of the spectrogram outputs: {Sxx.shape}/{f.shape}")
        return ds


if __name__ == "__main__":
    d = datetime(2023, 10, 14, 15, 56)
    r = IQStream(d)
    ds = r.to_pyfftw()
    p = AFRLPlots("14 Oct 2023 / PSD", date=datetime(2023, 10, 14, 15, 56))
    p.draw_psd(ds["fft"]["f_fft"] / 1e6, ds["fft"]["psd"], xlim=[0, 10])
    p.save("tmp/AFRL_psd.png")
    p.close()

    ds = r.to_spectrogram()
    print(ds["spectrogram"].keys())
    print(ds["spectrogram"]["t_spec"])
    p = AFRLPlots("14 Oct 2023 / PSD", date=datetime(2023, 10, 14, 15, 56))
    p.draw_psd_scan(
        ds["spectrogram"]["f"] / 1e6,
        ds["spectrogram"]["t_spec"],
        ds["spectrogram"]["psd"],
        xlim=[0, 10],
    )
    p.save("tmp/AFRL_psd_scan.png")
    p.close()
