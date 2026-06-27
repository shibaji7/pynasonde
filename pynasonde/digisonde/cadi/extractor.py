"""High-level CADI extractor helpers.

Phase-2 scope:
- decode MD2/MD4 binary files
- expose raw per-detection I/Q tables
- expose derived signal products (power, phase, Doppler-bin)
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pynasonde.digisonde.cadi.echo import (
    CadiArray,
    CadiEchoExtractor,
    CadiInterferometryExtractor,
    CadiReceiverLayout,
    compute_height_correction_km,
)
from pynasonde.digisonde.digi_utils import load_files_to_dataframe
from pynasonde.digisonde.cadi.reader import CadiReader


class CadiExtractor:
    """Extractor wrapper around :class:`CadiReader`."""

    def __init__(self, filename: str, dheight_km: float = 3.0):
        self.filename = filename
        self.reader = CadiReader(filename=filename, dheight_km=dheight_km)
        self.dataset = None

    @staticmethod
    def _as_signed(value: int) -> int:
        return value - 256 if value > 127 else value

    @staticmethod
    def _baseline_pairs(rx_indices: Sequence[int]) -> List[Tuple[int, int]]:
        pairs: List[Tuple[int, int]] = []
        for i in range(len(rx_indices)):
            for j in range(i + 1, len(rx_indices)):
                pairs.append((rx_indices[i], rx_indices[j]))
        return pairs

    @staticmethod
    def _group_coherence(
        z1: pd.Series, z2: pd.Series, group_index: pd.Series
    ) -> pd.Series:
        """Compute per-group complex coherence magnitude for one baseline."""
        out = pd.Series(index=z1.index, dtype=float)
        for _, idx in group_index.groupby(group_index).groups.items():
            v1 = z1.loc[idx].to_numpy(dtype=complex)
            v2 = z2.loc[idx].to_numpy(dtype=complex)
            num = np.abs(np.sum(v1 * np.conjugate(v2)))
            den = np.sqrt(np.sum(np.abs(v1) ** 2) * np.sum(np.abs(v2) ** 2))
            coh = float(num / den) if den > 0 else np.nan
            out.loc[idx] = coh
        return out

    def extract(self):
        """Parse the CADI file and cache the decoded dataset."""
        self.dataset = self.reader.parse()
        return self.dataset

    def to_dataframe_raw(self) -> pd.DataFrame:
        """Return one row per detected echo with dynamic per-RX I/Q columns."""
        if self.dataset is None:
            self.extract()

        rows = []
        freqs_hz = self.dataset.frequencies_hz
        dheight_km = self.dataset.dheight_km

        for det in self.dataset.detections:
            freq_hz = freqs_hz[det.frequency_index]
            row = {
                "site": self.dataset.header.site.strip(),
                "source_file": self.filename,
                "filetype": self.dataset.header.filetype,
                "time_index": det.time_index,
                "time_min": det.time_min,
                "time_sec": det.time_sec,
                "record_datetime": det.record_datetime,
                "frequency_index": det.frequency_index,
                "frequency_hz": float(freq_hz),
                "frequency_mhz": float(freq_hz) / 1e6,
                "noise_flag": det.noise_flag,
                "noise_power10": det.noise_power10,
                "gain_flag": det.gain_flag,
                "height_flag": det.height_flag,
                "height_km": float(det.height_flag) * dheight_km,
                "doppler_flag": det.doppler_flag,
            }

            for rec_idx, (i_raw, q_raw) in enumerate(det.iq_samples, start=1):
                row[f"rx{rec_idx}_i_raw"] = int(i_raw)
                row[f"rx{rec_idx}_q_raw"] = int(q_raw)
                row[f"rx{rec_idx}_i"] = self._as_signed(int(i_raw))
                row[f"rx{rec_idx}_q"] = self._as_signed(int(q_raw))

            rows.append(row)

        return pd.DataFrame(rows)

    def to_dataframe_products(
        self,
        coherence_groupby: Optional[Sequence[str]] = None,
        transmitting_delay_us: Optional[float] = None,
        sampling_delay_us: Optional[float] = None,
    ) -> pd.DataFrame:
        """Return per-detection table with derived power/phase fields.

        Derived columns:
            - ``doppler_bin``: integer Doppler-bin index (from ``doppler_flag``)
            - ``rxN_amp``: receiver amplitude ``sqrt(I^2 + Q^2)``
            - ``rxN_phase_rad`` / ``rxN_phase_deg``: per-receiver phase
            - ``mean_power_db``: ``20*log10(mean(rxN_amp))`` over non-empty channels
            - baseline phase differences ``dphi_AB_*`` and coherence ``coh_AB``

        Args:
            coherence_groupby:
                Grouping columns used for coherence estimation. Defaults to
                ``["time_index", "frequency_index"]``.
            transmitting_delay_us, sampling_delay_us:
                Optional CADI site timing delays from ``location.ini``. When
                both are supplied, ``height_km`` is corrected by
                ``0.5*c*(sampling_delay_us - transmitting_delay_us)`` and the
                original height is retained as ``height_uncorrected_km``.
        """
        df = self.to_dataframe_raw()
        if df.empty:
            return df

        height_correction_km = compute_height_correction_km(
            transmitting_delay_us=transmitting_delay_us,
            sampling_delay_us=sampling_delay_us,
        )
        df["height_uncorrected_km"] = df["height_km"]
        df["height_correction_km"] = height_correction_km
        df["transmitting_delay_us"] = (
            np.nan if transmitting_delay_us is None else float(transmitting_delay_us)
        )
        df["sampling_delay_us"] = (
            np.nan if sampling_delay_us is None else float(sampling_delay_us)
        )
        if np.isfinite(height_correction_km):
            df["height_km"] = df["height_uncorrected_km"] + height_correction_km

        if coherence_groupby is None:
            coherence_groupby = ("time_index", "frequency_index")

        df["doppler_bin"] = df["doppler_flag"].astype(int)

        rx_indices: List[int] = []
        for col in df.columns:
            if col.startswith("rx") and col.endswith("_i") and col[2:-2].isdigit():
                rx_indices.append(int(col[2:-2]))
        rx_indices = sorted(set(rx_indices))

        amp_cols: List[str] = []
        for idx in rx_indices:
            i_col = f"rx{idx}_i"
            q_col = f"rx{idx}_q"
            amp_col = f"rx{idx}_amp"
            phase_rad_col = f"rx{idx}_phase_rad"
            phase_deg_col = f"rx{idx}_phase_deg"

            df[amp_col] = np.sqrt(df[i_col].astype(float) ** 2 + df[q_col].astype(float) ** 2)
            df[phase_rad_col] = np.arctan2(df[q_col].astype(float), df[i_col].astype(float))
            df[phase_deg_col] = np.degrees(df[phase_rad_col])
            amp_cols.append(amp_col)

        if amp_cols:
            mean_amp = df[amp_cols].mean(axis=1)
            # Match legacy converter behavior: zero mean-amplitude maps to 0 dB.
            df["mean_power_db"] = np.where(
                mean_amp > 0.0, 20.0 * np.log10(mean_amp), 0.0
            )
        else:
            df["mean_power_db"] = np.nan

        # Interferometry-ready baseline products from per-RX phases.
        if len(rx_indices) >= 2:
            baselines = self._baseline_pairs(rx_indices)
            group_key = df[list(coherence_groupby)].astype(str).agg("|".join, axis=1)

            for a, b in baselines:
                pa = df[f"rx{a}_phase_rad"].to_numpy(dtype=float)
                pb = df[f"rx{b}_phase_rad"].to_numpy(dtype=float)
                dphi = np.angle(np.exp(1j * (pa - pb)))
                df[f"dphi_{a}{b}_rad"] = dphi
                df[f"dphi_{a}{b}_deg"] = np.degrees(dphi)

                za = df[f"rx{a}_i"].to_numpy(dtype=float) + 1j * df[
                    f"rx{a}_q"
                ].to_numpy(dtype=float)
                zb = df[f"rx{b}_i"].to_numpy(dtype=float) + 1j * df[
                    f"rx{b}_q"
                ].to_numpy(dtype=float)
                coh = self._group_coherence(
                    pd.Series(za, index=df.index),
                    pd.Series(zb, index=df.index),
                    group_key,
                )
                df[f"coh_{a}{b}"] = coh

        return df

    def to_dataframe_echoes(
        self,
        lat: float,
        lon: float,
        diagonal_m: float,
        orientation_deg: float = 0.0,
        rx_bitmask: Optional[int] = None,
        receiver_layout: Union[CadiReceiverLayout, str] = CadiReceiverLayout.STANDARD,
        unit_vectors: Optional[Mapping[int, Tuple[float, float]]] = None,
        dipole_orient_deg: Optional[Mapping[int, float]] = None,
        fft_size: Optional[int] = None,
        pulse_rate_hz: Optional[float] = None,
        transmitting_delay_us: Optional[float] = None,
        sampling_delay_us: Optional[float] = None,
        aoa_safe: bool = True,
        ambiguous_threshold_deg: float = 15.0,
        coherence_groupby: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Return CADI echo/interferometry products.

        Echo extraction requires at least three active receivers. Doppler Hz
        and line-of-sight velocity are returned as NaN unless ``fft_size`` and
        ``pulse_rate_hz`` are provided.
        """
        if self.dataset is None:
            self.extract()
        n_receivers = self.dataset.header.noofreceivers
        array = CadiArray.from_receiver_count(
            n_receivers=n_receivers,
            lat=lat,
            lon=lon,
            diagonal_m=diagonal_m,
            orientation_deg=orientation_deg,
            rx_bitmask=rx_bitmask,
            receiver_layout=receiver_layout,
            unit_vectors=unit_vectors,
            dipole_orient_deg=dipole_orient_deg,
        )
        array.validate_rx_count(n_receivers, min_receivers=3)

        products = self.to_dataframe_products(
            coherence_groupby=coherence_groupby,
            transmitting_delay_us=transmitting_delay_us,
            sampling_delay_us=sampling_delay_us,
        )
        return (
            CadiEchoExtractor(
                products,
                array=array,
                fft_size=fft_size,
                pulse_rate_hz=pulse_rate_hz,
                aoa_safe=aoa_safe,
                ambiguous_threshold_deg=ambiguous_threshold_deg,
            )
            .extract()
            .to_dataframe()
        )

    def to_dataframe_interferometry(
        self,
        lat: float,
        lon: float,
        diagonal_m: float,
        orientation_deg: float = 0.0,
        rx_bitmask: Optional[int] = None,
        receiver_layout: Union[CadiReceiverLayout, str] = CadiReceiverLayout.STANDARD,
        unit_vectors: Optional[Mapping[int, Tuple[float, float]]] = None,
        dipole_orient_deg: Optional[Mapping[int, float]] = None,
        transmitting_delay_us: Optional[float] = None,
        sampling_delay_us: Optional[float] = None,
        coherence_groupby: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Return baseline-level interferometry products.

        This product requires at least two active receivers. With two receivers,
        the output is a one-dimensional phase/projection estimate per baseline,
        not a full two-dimensional AOA/O-X echo solution.
        """
        if self.dataset is None:
            self.extract()
        n_receivers = self.dataset.header.noofreceivers
        array = CadiArray.from_receiver_count(
            n_receivers=n_receivers,
            lat=lat,
            lon=lon,
            diagonal_m=diagonal_m,
            orientation_deg=orientation_deg,
            rx_bitmask=rx_bitmask,
            receiver_layout=receiver_layout,
            unit_vectors=unit_vectors,
            dipole_orient_deg=dipole_orient_deg,
        )
        array.validate_rx_count(n_receivers, min_receivers=2)

        products = self.to_dataframe_products(
            coherence_groupby=coherence_groupby,
            transmitting_delay_us=transmitting_delay_us,
            sampling_delay_us=sampling_delay_us,
        )
        return CadiInterferometryExtractor(products, array=array).extract().to_dataframe()

    @staticmethod
    def extract_CADI(
        file: str,
        dheight_km: float = 3.0,
        product: str = "raw",
        coherence_groupby: Optional[Sequence[str]] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        diagonal_m: Optional[float] = None,
        orientation_deg: float = 0.0,
        rx_bitmask: Optional[int] = None,
        receiver_layout: Union[CadiReceiverLayout, str] = CadiReceiverLayout.STANDARD,
        unit_vectors: Optional[Mapping[int, Tuple[float, float]]] = None,
        dipole_orient_deg: Optional[Mapping[int, float]] = None,
        fft_size: Optional[int] = None,
        pulse_rate_hz: Optional[float] = None,
        transmitting_delay_us: Optional[float] = None,
        sampling_delay_us: Optional[float] = None,
        aoa_safe: bool = True,
        ambiguous_threshold_deg: float = 15.0,
    ) -> pd.DataFrame:
        """Extract one CADI file to a per-detection DataFrame.

        Args:
            product: ``"raw"``, ``"products"``, ``"interferometry"``, or
                ``"echoes"``.
        """
        extractor = CadiExtractor(file, dheight_km=dheight_km)
        if product == "raw":
            return extractor.to_dataframe_raw()
        if product == "products":
            return extractor.to_dataframe_products(
                coherence_groupby=coherence_groupby,
                transmitting_delay_us=transmitting_delay_us,
                sampling_delay_us=sampling_delay_us,
            )
        if product == "interferometry":
            if lat is None or lon is None or diagonal_m is None:
                raise ValueError(
                    "product='interferometry' requires lat, lon, and diagonal_m."
                )
            return extractor.to_dataframe_interferometry(
                lat=lat,
                lon=lon,
                diagonal_m=diagonal_m,
                orientation_deg=orientation_deg,
                rx_bitmask=rx_bitmask,
                receiver_layout=receiver_layout,
                unit_vectors=unit_vectors,
                dipole_orient_deg=dipole_orient_deg,
                transmitting_delay_us=transmitting_delay_us,
                sampling_delay_us=sampling_delay_us,
                coherence_groupby=coherence_groupby,
            )
        if product == "echoes":
            if lat is None or lon is None or diagonal_m is None:
                raise ValueError(
                    "product='echoes' requires lat, lon, and diagonal_m."
                )
            return extractor.to_dataframe_echoes(
                lat=lat,
                lon=lon,
                diagonal_m=diagonal_m,
                orientation_deg=orientation_deg,
                rx_bitmask=rx_bitmask,
                receiver_layout=receiver_layout,
                unit_vectors=unit_vectors,
                dipole_orient_deg=dipole_orient_deg,
                fft_size=fft_size,
                pulse_rate_hz=pulse_rate_hz,
                transmitting_delay_us=transmitting_delay_us,
                sampling_delay_us=sampling_delay_us,
                aoa_safe=aoa_safe,
                ambiguous_threshold_deg=ambiguous_threshold_deg,
                coherence_groupby=coherence_groupby,
            )
        raise ValueError(
            "product must be one of: raw, products, interferometry, echoes"
        )

    @staticmethod
    def load_CADI_files(
        folders: Iterable[str],
        exts: Union[str, Sequence[str]] = ("*.md2", "*.md4", "*.MD2", "*.MD4"),
        n_procs: int = 1,
        dheight_km: float = 3.0,
        product: str = "raw",
        coherence_groupby: Optional[Sequence[str]] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        diagonal_m: Optional[float] = None,
        orientation_deg: float = 0.0,
        rx_bitmask: Optional[int] = None,
        receiver_layout: Union[CadiReceiverLayout, str] = CadiReceiverLayout.STANDARD,
        unit_vectors: Optional[Mapping[int, Tuple[float, float]]] = None,
        dipole_orient_deg: Optional[Mapping[int, float]] = None,
        fft_size: Optional[int] = None,
        pulse_rate_hz: Optional[float] = None,
        transmitting_delay_us: Optional[float] = None,
        sampling_delay_us: Optional[float] = None,
        aoa_safe: bool = True,
        ambiguous_threshold_deg: float = 15.0,
    ) -> pd.DataFrame:
        """Batch-load CADI files from one or more folders.

        Args:
            folders: Iterable of folders to search.
            exts: One glob or a sequence of globs.
            n_procs: Worker count (``1`` uses sequential processing).
            dheight_km: Height-bin spacing in km.
            product: ``"raw"``, ``"products"``, ``"interferometry"``, or
                ``"echoes"``.
            coherence_groupby: Grouping columns for interferometric coherence.
            lat: Station latitude, required for ``product="echoes"``.
            lon: Station longitude, required for ``product="echoes"``.
            diagonal_m: CADI array diagonal in meters, required for echoes.
            transmitting_delay_us, sampling_delay_us: Optional CADI timing
                delays. When both are supplied, products/interferometry/echoes
                use corrected ``height_km`` and retain
                ``height_uncorrected_km``.
        """
        return load_files_to_dataframe(
            folders=folders,
            exts=exts,
            extractor=CadiExtractor.extract_CADI,
            n_procs=n_procs,
            extractor_kwargs=dict(
                dheight_km=dheight_km,
                product=product,
                coherence_groupby=coherence_groupby,
                lat=lat,
                lon=lon,
                diagonal_m=diagonal_m,
                orientation_deg=orientation_deg,
                rx_bitmask=rx_bitmask,
                receiver_layout=receiver_layout,
                unit_vectors=unit_vectors,
                dipole_orient_deg=dipole_orient_deg,
                fft_size=fft_size,
                pulse_rate_hz=pulse_rate_hz,
                transmitting_delay_us=transmitting_delay_us,
                sampling_delay_us=sampling_delay_us,
                aoa_safe=aoa_safe,
                ambiguous_threshold_deg=ambiguous_threshold_deg,
            ),
        )
