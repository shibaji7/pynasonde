"""CADI echo and interferometry helpers.

This module contains the CADI-specific layer above raw MD2/MD4 decoding:
array geometry, O/X mode classification, angle-of-arrival estimates, and
optional Doppler-bin calibration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

C_LIGHT = 299_792_458.0


def compute_height_correction_km(
    transmitting_delay_us: Optional[float] = None,
    sampling_delay_us: Optional[float] = None,
) -> float:
    """Return CADI virtual-height timing correction in km.

    The correction is only defined when both site timing delays are available.
    Positive values increase ``height_km``. For the common SHA location.ini
    values ``sampling_delay_us=2063`` and ``transmitting_delay_us=2056``, the
    correction is about 1.05 km.
    """
    if transmitting_delay_us is None or sampling_delay_us is None:
        return np.nan
    delay_us = float(sampling_delay_us) - float(transmitting_delay_us)
    return float(0.5 * C_LIGHT * delay_us * 1e-6 / 1000.0)


class CadiReceiverLayout(str, Enum):
    """Named CADI receiver-layout presets.

    Use ``STANDARD`` for the nominal CADI diamond. Use ``CUSTOM`` with
    explicit ``unit_vectors`` and/or ``dipole_orient_deg`` when receiver
    numbering or antenna orientation differs by site.
    """

    STANDARD = "standard"
    CUSTOM = "custom"

    @classmethod
    def coerce(cls, layout: Union["CadiReceiverLayout", str]) -> "CadiReceiverLayout":
        if isinstance(layout, cls):
            return layout
        return cls(str(layout).lower())


CADI_STANDARD_UNIT_VECTORS: Dict[int, Tuple[float, float]] = {
    1: (0.0, 1.0),
    2: (1.0, 0.0),
    3: (0.0, -1.0),
    4: (-1.0, 0.0),
}
CADI_STANDARD_DIPOLE_ORIENT_DEG: Dict[int, float] = {
    1: 0.0,
    2: 90.0,
    3: 0.0,
    4: 90.0,
}


@dataclass
class CadiArray:
    """CADI diamond receiver array geometry."""

    lat: float
    lon: float
    diagonal_m: float
    orientation_deg: float = 0.0
    rx_bitmask: int = 15
    receiver_layout: Union[CadiReceiverLayout, str] = CadiReceiverLayout.STANDARD
    unit_vectors: Optional[Mapping[int, Tuple[float, float]]] = None
    dipole_orient_deg: Optional[Mapping[int, float]] = None

    def __post_init__(self) -> None:
        self.half_diag = self.diagonal_m / 2.0
        self.side_m = self.diagonal_m / np.sqrt(2.0)
        self.hemisphere = "N" if self.lat >= 0 else "S"
        self.receiver_layout = CadiReceiverLayout.coerce(self.receiver_layout)
        self.unit_vectors, self.dipole_orient_deg = self._resolve_layout()

        rot_rad = np.deg2rad(self.orientation_deg)
        cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
        self.rx_positions: Dict[int, Tuple[float, float]] = {}
        self.dipole_orientations: Dict[int, float] = {}

        for rx_id, (ue, un) in self.unit_vectors.items():
            e0 = ue * self.half_diag
            n0 = un * self.half_diag
            east = e0 * cos_r - n0 * sin_r
            north = e0 * sin_r + n0 * cos_r
            self.rx_positions[rx_id] = (east, north)
            self.dipole_orientations[rx_id] = (
                self.dipole_orient_deg[rx_id] + self.orientation_deg
            ) % 180.0

        self.active_rx = [i + 1 for i in range(4) if self.rx_bitmask & (1 << i)]
        self.n_rx = len(self.active_rx)
        self.baselines = self._build_baselines()

    def _resolve_layout(self) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, float]]:
        if self.receiver_layout == CadiReceiverLayout.CUSTOM:
            if self.unit_vectors is None or self.dipole_orient_deg is None:
                raise ValueError(
                    "receiver_layout='custom' requires unit_vectors and "
                    "dipole_orient_deg."
                )
            unit_vectors = dict(self.unit_vectors)
            dipole_orient_deg = dict(self.dipole_orient_deg)
        else:
            unit_vectors = dict(CADI_STANDARD_UNIT_VECTORS)
            dipole_orient_deg = dict(CADI_STANDARD_DIPOLE_ORIENT_DEG)
            if self.unit_vectors is not None:
                unit_vectors.update(dict(self.unit_vectors))
            if self.dipole_orient_deg is not None:
                dipole_orient_deg.update(dict(self.dipole_orient_deg))

        expected_rx = {1, 2, 3, 4}
        if set(unit_vectors) != expected_rx:
            raise ValueError("unit_vectors must define receiver IDs 1, 2, 3, and 4.")
        if set(dipole_orient_deg) != expected_rx:
            raise ValueError(
                "dipole_orient_deg must define receiver IDs 1, 2, 3, and 4."
            )
        return unit_vectors, dipole_orient_deg

    @classmethod
    def from_receiver_count(
        cls,
        n_receivers: int,
        lat: float,
        lon: float,
        diagonal_m: float,
        orientation_deg: float = 0.0,
        rx_bitmask: Optional[int] = None,
        receiver_layout: Union[CadiReceiverLayout, str] = CadiReceiverLayout.STANDARD,
        unit_vectors: Optional[Mapping[int, Tuple[float, float]]] = None,
        dipole_orient_deg: Optional[Mapping[int, float]] = None,
    ) -> "CadiArray":
        """Build geometry from file receiver count and optional active bitmask."""
        if rx_bitmask is None:
            rx_bitmask = (1 << n_receivers) - 1
        return cls(
            lat=lat,
            lon=lon,
            diagonal_m=diagonal_m,
            orientation_deg=orientation_deg,
            rx_bitmask=rx_bitmask,
            receiver_layout=receiver_layout,
            unit_vectors=unit_vectors,
            dipole_orient_deg=dipole_orient_deg,
        )

    def _build_baselines(self) -> List[Tuple[int, int, float, float]]:
        baselines = []
        for i, rx_i in enumerate(self.active_rx):
            for rx_j in self.active_rx[i + 1 :]:
                ei, ni = self.rx_positions[rx_i]
                ej, nj = self.rx_positions[rx_j]
                baselines.append((rx_i, rx_j, ej - ei, nj - ni))
        return baselines

    def validate_rx_count(self, n_rx_from_file: int, min_receivers: int = 3) -> None:
        """Validate active/file receiver counts for echo extraction."""
        if n_rx_from_file < min_receivers:
            raise ValueError(
                f"CADI echo extraction requires at least {min_receivers} receivers; "
                f"file reports {n_rx_from_file}."
            )
        if self.n_rx < min_receivers:
            raise ValueError(
                f"CADI echo extraction requires at least {min_receivers} active "
                f"receivers; rx_bitmask={self.rx_bitmask} enables {self.n_rx}."
            )
        if self.n_rx != n_rx_from_file:
            raise ValueError(
                f"rx_bitmask={self.rx_bitmask} implies {self.n_rx} receivers, "
                f"but file header reports {n_rx_from_file}."
            )

    def get_perpendicular_pairs(self) -> List[Tuple[int, int]]:
        pairs = []
        for i, rx_a in enumerate(self.active_rx):
            for rx_b in self.active_rx[i + 1 :]:
                diff = abs(
                    self.dipole_orientations[rx_a] - self.dipole_orientations[rx_b]
                )
                diff = diff % 180.0
                if abs(diff - 90.0) < 15.0:
                    pairs.append((rx_a, rx_b))
        return pairs

    def get_same_orientation_baselines(self) -> List[Tuple[int, int, float, float]]:
        same = []
        for rx_i, rx_j, de, dn in self.baselines:
            diff = abs(
                self.dipole_orientations[rx_i] - self.dipole_orientations[rx_j]
            ) % 180.0
            if diff < 15.0:
                same.append((rx_i, rx_j, de, dn))
        return same

    def describe(self) -> dict:
        """Return a debug summary of receiver geometry and baselines."""
        receivers = []
        for rx_id in sorted(self.rx_positions):
            east, north = self.rx_positions[rx_id]
            receivers.append(
                {
                    "rx": rx_id,
                    "active": rx_id in self.active_rx,
                    "east_m": float(east),
                    "north_m": float(north),
                    "dipole_orient_deg": float(self.dipole_orientations[rx_id]),
                }
            )

        baselines = []
        same = set(self.get_same_orientation_baselines())
        for rx_i, rx_j, de, dn in self.baselines:
            baselines.append(
                {
                    "rx_i": rx_i,
                    "rx_j": rx_j,
                    "delta_east_m": float(de),
                    "delta_north_m": float(dn),
                    "baseline_m": float(np.hypot(de, dn)),
                    "same_orientation": (rx_i, rx_j, de, dn) in same,
                }
            )

        return {
            "receiver_layout": self.receiver_layout.value,
            "orientation_deg": float(self.orientation_deg),
            "rx_bitmask": int(self.rx_bitmask),
            "active_rx": list(self.active_rx),
            "receivers": receivers,
            "baselines": baselines,
            "perpendicular_pairs": self.get_perpendicular_pairs(),
        }


@dataclass
class CadiEcho:
    """One CADI echo with interferometric products."""

    frequency_hz: float = np.nan
    frequency_mhz: float = np.nan
    height_km: float = np.nan
    height_uncorrected_km: float = np.nan
    height_correction_km: float = np.nan
    transmitting_delay_us: float = np.nan
    sampling_delay_us: float = np.nan
    mean_power_db: float = np.nan
    doppler_bin: int = -1
    doppler_hz: float = np.nan
    v_los: float = np.nan
    XL: float = np.nan
    YL: float = np.nan
    ZL: float = np.nan
    zenith: float = np.nan
    azimuth: float = np.nan
    aoa_residual: float = np.nan
    aoa_method: str = "unknown"
    polarization_deg: float = np.nan
    mode: str = "unknown"
    rx_count: int = 0
    source_file: str = ""
    record_datetime: object = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CadiInterferometryProduct:
    """One CADI baseline-level interferometry product."""

    frequency_hz: float = np.nan
    frequency_mhz: float = np.nan
    height_km: float = np.nan
    height_uncorrected_km: float = np.nan
    height_correction_km: float = np.nan
    transmitting_delay_us: float = np.nan
    sampling_delay_us: float = np.nan
    mean_power_db: float = np.nan
    doppler_bin: int = -1
    rx_a: int = 0
    rx_b: int = 0
    baseline_m: float = np.nan
    baseline_azimuth: float = np.nan
    phase_rad: float = np.nan
    phase_deg: float = np.nan
    coherence: float = np.nan
    projected_direction_cosine: float = np.nan
    projected_angle_deg: float = np.nan
    rx_count: int = 0
    source_file: str = ""
    record_datetime: object = None

    def to_dict(self) -> dict:
        return asdict(self)


def doppler_frequency(
    bin_index: int,
    fft_size: Optional[int],
    pulse_rate_hz: Optional[float],
) -> float:
    """Convert Doppler bin index to Hz when calibration is available."""
    if fft_size is None or pulse_rate_hz is None:
        return np.nan
    return float((bin_index - fft_size // 2) * (pulse_rate_hz / fft_size))


def los_velocity(fd_hz: float, freq_hz: float) -> float:
    """Line-of-sight velocity from Doppler frequency."""
    if not np.isfinite(fd_hz) or freq_hz <= 0:
        return np.nan
    return float(fd_hz * (C_LIGHT / freq_hz) / 2.0)


def map_file_iq_to_physical_rx(row: pd.Series, array: CadiArray) -> Dict[int, complex]:
    """Map file-order receiver columns to physical CADI receiver IDs.

    CADI blocks store only active receiver samples in order. If ``rx_bitmask``
    is 14, for example, file columns ``rx1_*``, ``rx2_*``, ``rx3_*`` represent
    physical receivers Rx2, Rx3, and Rx4 respectively.
    """
    iq = {}
    for file_idx, physical_rx in enumerate(array.active_rx, start=1):
        i_col = f"rx{file_idx}_i"
        q_col = f"rx{file_idx}_q"
        if i_col in row.index and q_col in row.index:
            iq[physical_rx] = complex(float(row[i_col]), float(row[q_col]))
    return iq


def classify_ox_mode(
    iq_data: Dict[int, complex],
    array: CadiArray,
    ambiguous_threshold_deg: float = 15.0,
) -> Tuple[str, float]:
    """Classify CADI echo O/X mode from perpendicular dipole phase."""
    phase_votes = []
    for rx_a, rx_b in array.get_perpendicular_pairs():
        if rx_a not in iq_data or rx_b not in iq_data:
            continue

        orient_a = array.dipole_orientations[rx_a]
        orient_b = array.dipole_orientations[rx_b]
        if abs(orient_a - 90.0) < abs(orient_b - 90.0):
            iq_ew, iq_ns = iq_data[rx_a], iq_data[rx_b]
        else:
            iq_ew, iq_ns = iq_data[rx_b], iq_data[rx_a]

        phase_votes.append(np.angle(iq_ns * np.conjugate(iq_ew)))

    if not phase_votes:
        return "unknown", np.nan

    pp_rad = float(np.angle(np.mean(np.exp(1j * np.array(phase_votes)))))
    pp_deg = float(np.rad2deg(pp_rad))
    if abs(pp_deg) < ambiguous_threshold_deg:
        return "ambiguous", pp_deg

    is_lhc = pp_rad > 0
    if array.hemisphere == "N":
        return ("O" if is_lhc else "X"), pp_deg
    return ("O" if not is_lhc else "X"), pp_deg


def compute_aoa(
    iq_data: Dict[int, complex],
    array: CadiArray,
    freq_hz: float,
    aoa_safe: bool = True,
) -> dict:
    """Compute CADI direction cosines and arrival angles."""
    same_bl = array.get_same_orientation_baselines()
    k = 2.0 * np.pi * freq_hz / C_LIGHT

    if aoa_safe and len(same_bl) >= 2:
        baselines = same_bl
        method = "same_orient"
    else:
        baselines = array.baselines
        method = "uncorrected" if not same_bl else "fallback_all_baselines"

    if len(baselines) < 2:
        raise ValueError(f"Need at least 2 baselines for 2D AOA; have {len(baselines)}.")

    A_rows = []
    b_vals = []
    for rx_i, rx_j, de, dn in baselines:
        if rx_i not in iq_data or rx_j not in iq_data:
            continue
        A_rows.append([de * k, dn * k])
        b_vals.append(np.angle(iq_data[rx_j] * np.conjugate(iq_data[rx_i])))

    if len(A_rows) < 2:
        raise ValueError("Not enough valid receiver baselines for AOA.")

    A = np.array(A_rows)
    b = np.array(b_vals)
    result = np.linalg.lstsq(A, b, rcond=None)
    xl, yl = result[0]
    residual = float(result[1][0]) if len(result[1]) > 0 else 0.0

    r2 = float(xl**2 + yl**2)
    if r2 > 1.0:
        norm = np.sqrt(r2)
        xl /= norm
        yl /= norm
        r2 = 1.0
    zl = float(np.sqrt(max(0.0, 1.0 - r2)))

    zenith = float(np.rad2deg(np.arccos(np.clip(zl, -1.0, 1.0))))
    azimuth = float(np.rad2deg(np.arctan2(xl, yl)))
    if azimuth < 0:
        azimuth += 360.0

    return {
        "XL": float(xl),
        "YL": float(yl),
        "ZL": zl,
        "zenith": zenith,
        "azimuth": azimuth,
        "residual": residual,
        "method": method,
    }


def debug_echo_geometry(
    row: pd.Series,
    array: CadiArray,
    freq_hz: Optional[float] = None,
    aoa_safe: bool = True,
    ambiguous_threshold_deg: float = 15.0,
) -> dict:
    """Return receiver mapping, baseline phases, O/X, and AOA for one echo row."""
    if freq_hz is None:
        freq_hz = float(row["frequency_hz"])

    iq = map_file_iq_to_physical_rx(row, array)
    receiver_mapping = []
    for file_idx, physical_rx in enumerate(array.active_rx, start=1):
        receiver_mapping.append(
            {
                "file_rx": file_idx,
                "physical_rx": physical_rx,
                "i_col": f"rx{file_idx}_i",
                "q_col": f"rx{file_idx}_q",
                "has_data": physical_rx in iq,
            }
        )

    baseline_phases = []
    for rx_i, rx_j, de, dn in array.baselines:
        if rx_i not in iq or rx_j not in iq:
            continue
        phase_rad = float(np.angle(iq[rx_j] * np.conjugate(iq[rx_i])))
        same_orientation = (
            abs(array.dipole_orientations[rx_i] - array.dipole_orientations[rx_j])
            % 180.0
        ) < 15.0
        baseline_phases.append(
            {
                "rx_i": rx_i,
                "rx_j": rx_j,
                "delta_east_m": float(de),
                "delta_north_m": float(dn),
                "phase_rad": phase_rad,
                "phase_deg": float(np.rad2deg(phase_rad)),
                "same_orientation": same_orientation,
            }
        )

    mode, polarization_deg = classify_ox_mode(
        iq, array, ambiguous_threshold_deg=ambiguous_threshold_deg
    )
    try:
        aoa = compute_aoa(iq, array, freq_hz=freq_hz, aoa_safe=aoa_safe)
    except (ValueError, np.linalg.LinAlgError) as exc:
        aoa = {"method": "failed", "error": str(exc)}

    return {
        "array": array.describe(),
        "receiver_mapping": receiver_mapping,
        "baseline_phases": baseline_phases,
        "mode": mode,
        "polarization_deg": polarization_deg,
        "aoa": aoa,
    }


class CadiEchoExtractor:
    """Build echo-level products from a CADI product DataFrame."""

    def __init__(
        self,
        df: pd.DataFrame,
        array: CadiArray,
        fft_size: Optional[int] = None,
        pulse_rate_hz: Optional[float] = None,
        aoa_safe: bool = True,
        ambiguous_threshold_deg: float = 15.0,
    ) -> None:
        self.df = df
        self.array = array
        self.fft_size = fft_size
        self.pulse_rate_hz = pulse_rate_hz
        self.aoa_safe = aoa_safe
        self.ambiguous_threshold_deg = ambiguous_threshold_deg
        self.echoes: List[CadiEcho] = []

    def extract(self) -> "CadiEchoExtractor":
        echoes = []
        for _, row in self.df.iterrows():
            iq = map_file_iq_to_physical_rx(row, self.array)

            if len(iq) < 3:
                continue

            mode, pp_deg = classify_ox_mode(
                iq, self.array, ambiguous_threshold_deg=self.ambiguous_threshold_deg
            )
            try:
                aoa = compute_aoa(
                    iq,
                    self.array,
                    float(row["frequency_hz"]),
                    aoa_safe=self.aoa_safe,
                )
            except (ValueError, np.linalg.LinAlgError):
                aoa = {
                    "XL": np.nan,
                    "YL": np.nan,
                    "ZL": np.nan,
                    "zenith": np.nan,
                    "azimuth": np.nan,
                    "residual": np.nan,
                    "method": "failed",
                }

            fd = doppler_frequency(
                int(row["doppler_bin"]), self.fft_size, self.pulse_rate_hz
            )
            echoes.append(
                CadiEcho(
                    frequency_hz=float(row["frequency_hz"]),
                    frequency_mhz=float(row["frequency_mhz"]),
                    height_km=float(row["height_km"]),
                    height_uncorrected_km=float(
                        row.get("height_uncorrected_km", row["height_km"])
                    ),
                    height_correction_km=float(row.get("height_correction_km", np.nan)),
                    transmitting_delay_us=float(
                        row.get("transmitting_delay_us", np.nan)
                    ),
                    sampling_delay_us=float(row.get("sampling_delay_us", np.nan)),
                    mean_power_db=float(row.get("mean_power_db", np.nan)),
                    doppler_bin=int(row["doppler_bin"]),
                    doppler_hz=fd,
                    v_los=los_velocity(fd, float(row["frequency_hz"])),
                    XL=aoa["XL"],
                    YL=aoa["YL"],
                    ZL=aoa["ZL"],
                    zenith=aoa["zenith"],
                    azimuth=aoa["azimuth"],
                    aoa_residual=aoa["residual"],
                    aoa_method=aoa["method"],
                    polarization_deg=pp_deg,
                    mode=mode,
                    rx_count=len(iq),
                    source_file=str(row.get("source_file", "")),
                    record_datetime=row.get("record_datetime", None),
                )
            )

        self.echoes = echoes
        return self

    def to_dataframe(self) -> pd.DataFrame:
        if not self.echoes:
            return pd.DataFrame()
        return pd.DataFrame.from_records([echo.to_dict() for echo in self.echoes])


class CadiInterferometryExtractor:
    """Build baseline-level products from at least two CADI receivers."""

    def __init__(self, df: pd.DataFrame, array: CadiArray) -> None:
        self.df = df
        self.array = array
        self.products: List[CadiInterferometryProduct] = []

    def extract(self) -> "CadiInterferometryExtractor":
        products = []
        for _, row in self.df.iterrows():
            iq = map_file_iq_to_physical_rx(row, self.array)

            if len(iq) < 2:
                continue

            freq_hz = float(row["frequency_hz"])
            k = 2.0 * np.pi * freq_hz / C_LIGHT
            for rx_a, rx_b, de, dn in self.array.baselines:
                if rx_a not in iq or rx_b not in iq:
                    continue

                baseline_m = float(np.hypot(de, dn))
                phase_rad = float(np.angle(iq[rx_b] * np.conjugate(iq[rx_a])))
                projected = np.nan
                projected_angle = np.nan
                if baseline_m > 0 and k > 0:
                    projected = float(np.clip(phase_rad / (k * baseline_m), -1.0, 1.0))
                    projected_angle = float(np.rad2deg(np.arcsin(projected)))

                baseline_azimuth = float(np.rad2deg(np.arctan2(de, dn)))
                if baseline_azimuth < 0:
                    baseline_azimuth += 360.0

                products.append(
                    CadiInterferometryProduct(
                        frequency_hz=freq_hz,
                        frequency_mhz=float(row["frequency_mhz"]),
                        height_km=float(row["height_km"]),
                        height_uncorrected_km=float(
                            row.get("height_uncorrected_km", row["height_km"])
                        ),
                        height_correction_km=float(
                            row.get("height_correction_km", np.nan)
                        ),
                        transmitting_delay_us=float(
                            row.get("transmitting_delay_us", np.nan)
                        ),
                        sampling_delay_us=float(row.get("sampling_delay_us", np.nan)),
                        mean_power_db=float(row.get("mean_power_db", np.nan)),
                        doppler_bin=int(row["doppler_bin"]),
                        rx_a=rx_a,
                        rx_b=rx_b,
                        baseline_m=baseline_m,
                        baseline_azimuth=baseline_azimuth,
                        phase_rad=phase_rad,
                        phase_deg=float(np.rad2deg(phase_rad)),
                        coherence=float(row.get(f"coh_{rx_a}{rx_b}", np.nan)),
                        projected_direction_cosine=projected,
                        projected_angle_deg=projected_angle,
                        rx_count=len(iq),
                        source_file=str(row.get("source_file", "")),
                        record_datetime=row.get("record_datetime", None),
                    )
                )

        self.products = products
        return self

    def to_dataframe(self) -> pd.DataFrame:
        if not self.products:
            return pd.DataFrame()
        return pd.DataFrame.from_records([product.to_dict() for product in self.products])


def compute_velocity_from_skymap(skymap_points: Sequence[dict]) -> dict:
    """Fit 3D drift velocity from echo direction cosines and LOS velocities."""
    if len(skymap_points) < 3:
        raise ValueError(f"Need at least 3 sky map points; have {len(skymap_points)}.")
    A = np.array([[p["XL"], p["YL"], p["ZL"]] for p in skymap_points])
    b = np.array([p["v_los"] for p in skymap_points])
    result = np.linalg.lstsq(A, b, rcond=None)
    vx, vy, vz = result[0]
    fitted = A @ result[0]
    rms = float(np.sqrt(np.mean((b - fitted) ** 2)))
    vh = float(np.sqrt(vx**2 + vy**2))
    v_total = float(np.sqrt(vx**2 + vy**2 + vz**2))
    azimuth = float(np.rad2deg(np.arctan2(vx, vy)))
    if azimuth < 0:
        azimuth += 360.0
    return {
        "Vx": float(vx),
        "Vy": float(vy),
        "Vz": float(vz),
        "Vh": vh,
        "V_total": v_total,
        "azimuth": azimuth,
        "n_points": len(skymap_points),
        "residual": rms,
    }
