"""SZA-aware parameter presets for Digisonde ionogram scaling."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

from pynasonde.digisonde.scaler.solar import classify_solar_regime, solar_zenith_angle

SOLAR_PARAMETER_SETS: dict[str, dict] = {
    "day": {
        "dbscan_eps": 0.010,
        "dbscan_min_samples": 4,
        "min_amp_dB": 27.0,
        "E": {
            "h_lo": 100.0,
            "h_hi": 150.0,
            "sigma_trans_km": 8.0,
            "max_jump_km": 20.0,
            "f_lo": 2.0,
            "f_hi": 5.0,
        },
        "F2": {
            "h_lo": 180.0,
            "h_hi": 420.0,
            "sigma_trans_km": 25.0,
            "max_jump_km": 40.0,
            "f_lo": 2.0,
            "f_hi": 12.0,
        },
    },
    "twilight": {
        "dbscan_eps": 0.010,
        "dbscan_min_samples": 4,
        "min_amp_dB": 27.0,
        "E": {
            "h_lo": 100.0,
            "h_hi": 150.0,
            "sigma_trans_km": 10.0,
            "max_jump_km": 25.0,
            "f_lo": 2.0,
            "f_hi": 5.0,
        },
        "F2": {
            "h_lo": 220.0,
            "h_hi": 470.0,
            "sigma_trans_km": 35.0,
            "max_jump_km": 55.0,
            "f_lo": 2.0,
            "f_hi": 12.0,
        },
    },
    "night": {
        "dbscan_eps": 0.010,
        "dbscan_min_samples": 4,
        "min_amp_dB": 27.0,
        "E": {
            "h_lo": 100.0,
            "h_hi": 150.0,
            "sigma_trans_km": 12.0,
            "max_jump_km": 30.0,
            "f_lo": 2.0,
            "f_hi": 4.5,
        },
        "F2": {
            "h_lo": 250.0,
            "h_hi": 550.0,
            "sigma_trans_km": 50.0,
            "max_jump_km": 70.0,
            "f_lo": 2.0,
            "f_hi": 12.0,
        },
    },
}

BASE_LAYER_SCALER_KWARGS: dict[str, dict] = {
    "E": {
        "dbscan_eps": 0.035,
        "dbscan_min_samples": 10,
        "min_echoes": 3,
        "min_amp_dB": 12.0,
        "sigma_trans_km": 10.0,
        "consecutive_gap": 8,
        "max_jump_km": 25.0,
    },
    "F2": {
        "dbscan_eps": 0.02,
        "dbscan_min_samples": 5,
        "min_echoes": 5,
        "min_amp_dB": 24.0,
        "sigma_trans_km": 18.0,
        "max_jump_km": 22.0,
    },
}

TUNING_PRESETS: dict[str, dict] = {
    "balanced": {
        "description": "Balanced SZA-driven parameters.",
        "E": {},
        "F2": {},
    },
    "e_sensitive": {
        "description": "More permissive E-region detection for weak echoes.",
        "E": {
            "scaler_kwargs": {
                "dbscan_eps": 0.020,
                "dbscan_min_samples": 3,
                "min_amp_dB": 18.0,
                "min_echoes": 2,
                "consecutive_gap": 10,
                "max_jump_km": 28.0,
                "sigma_trans_km": 12.0,
            },
            "layer": {
                "h_lo": 90.0,
                "h_hi": 170.0,
                "f_lo": 1.5,
                "f_hi": 5.5,
                "sigma_trans_km": 12.0,
                "max_jump_km": 28.0,
            },
        },
        "F2": {},
    },
    "f2_smooth": {
        "description": "More continuous F2 tracing with stronger smoothing.",
        "E": {},
        "F2": {
            "scaler_kwargs": {
                "dbscan_eps": 0.015,
                "dbscan_min_samples": 5,
                "min_amp_dB": 25.0,
                "min_echoes": 5,
                "sigma_trans_km": 28.0,
                "max_jump_km": 45.0,
            },
            "layer": {
                "sigma_trans_km": 32.0,
                "max_jump_km": 50.0,
            },
        },
    },
    "strict": {
        "description": "Higher-confidence traces for noisy ionograms.",
        "E": {
            "scaler_kwargs": {
                "dbscan_eps": 0.010,
                "dbscan_min_samples": 5,
                "min_amp_dB": 27.0,
                "min_echoes": 4,
                "max_jump_km": 18.0,
            },
            "layer": {
                "h_lo": 100.0,
                "h_hi": 155.0,
                "f_lo": 2.0,
                "f_hi": 5.0,
                "max_jump_km": 18.0,
            },
        },
        "F2": {
            "scaler_kwargs": {
                "dbscan_eps": 0.010,
                "dbscan_min_samples": 6,
                "min_amp_dB": 28.0,
                "min_echoes": 6,
                "sigma_trans_km": 20.0,
                "max_jump_km": 30.0,
            },
            "layer": {
                "sigma_trans_km": 22.0,
                "max_jump_km": 35.0,
            },
        },
    },
    "permissive": {
        "description": "Broad capture pass for diagnosing missed traces.",
        "E": {
            "scaler_kwargs": {
                "dbscan_eps": 0.030,
                "dbscan_min_samples": 3,
                "min_amp_dB": 16.0,
                "min_echoes": 2,
                "consecutive_gap": 12,
                "sigma_trans_km": 14.0,
                "max_jump_km": 32.0,
            },
            "layer": {
                "h_lo": 90.0,
                "h_hi": 180.0,
                "f_lo": 1.5,
                "f_hi": 6.0,
                "sigma_trans_km": 14.0,
                "max_jump_km": 32.0,
            },
        },
        "F2": {
            "scaler_kwargs": {
                "dbscan_eps": 0.025,
                "dbscan_min_samples": 4,
                "min_amp_dB": 22.0,
                "min_echoes": 4,
                "sigma_trans_km": 35.0,
                "max_jump_km": 60.0,
            },
            "layer": {
                "h_lo": 170.0,
                "h_hi": 560.0,
                "f_lo": 1.8,
                "f_hi": 13.0,
                "sigma_trans_km": 40.0,
                "max_jump_km": 65.0,
            },
        },
    },
}


@dataclass(frozen=True)
class ScalerPreset:
    """Effective layer and scaler settings for one ionogram."""

    regime: str
    sza_deg: float | None
    tuning_preset: str
    layers: dict[str, list[dict]]
    scaler_kwargs: dict[str, dict]
    enabled: dict[str, bool]
    metadata: dict[str, object] = field(default_factory=dict)

    def layer(self, name: str) -> list[dict]:
        """Return the one-layer config list for ``name``."""
        return self.layers[name]

    def kwargs(self, name: str) -> dict:
        """Return scaler kwargs for ``name``."""
        return self.scaler_kwargs[name]


def _valid_regime(regime: str) -> str:
    regime = regime.lower()
    if regime not in SOLAR_PARAMETER_SETS:
        raise ValueError(
            f"Unknown solar regime {regime!r}; expected one of "
            f"{sorted(SOLAR_PARAMETER_SETS)}."
        )
    return regime


def _valid_tuning_preset(tuning_preset: str) -> str:
    tuning_preset = tuning_preset.lower()
    if tuning_preset not in TUNING_PRESETS:
        raise ValueError(
            f"Unknown tuning preset {tuning_preset!r}; expected one of "
            f"{sorted(TUNING_PRESETS)}."
        )
    return tuning_preset


def _effective_e_f_hi(layer: dict, sza_deg: float | None) -> float:
    """Return E-layer ``f_hi`` after daylight SZA frequency limiting."""
    f_lo = float(layer["f_lo"])
    f_hi = float(layer["f_hi"])
    if sza_deg is not None and 0.0 < float(sza_deg) < 90.0:
        f_hi = min(f_hi, 6.0 * math.cos(math.radians(float(sza_deg))))
    if f_hi < f_lo:
        f_hi = f_lo + 2.0
    return f_hi


def _build_layer(
    name: str,
    regime_params: dict,
    tuning: dict,
    sza_deg: float | None,
) -> list[dict]:
    layer = dict(regime_params[name])
    layer.update(tuning.get(name, {}).get("layer", {}))
    layer["name"] = name
    if name == "E":
        layer["f_hi"] = _effective_e_f_hi(layer, sza_deg)
    return [layer]


def _build_scaler_kwargs(
    name: str,
    regime_params: dict,
    tuning: dict,
    rfi_filter: object | None,
) -> dict:
    kwargs = dict(BASE_LAYER_SCALER_KWARGS[name])
    kwargs.update(
        {
            "dbscan_eps": regime_params["dbscan_eps"],
            "dbscan_min_samples": regime_params["dbscan_min_samples"],
            "min_amp_dB": regime_params["min_amp_dB"],
        }
    )
    kwargs.update(tuning.get(name, {}).get("scaler_kwargs", {}))
    if rfi_filter is not None:
        kwargs["rfi_filter"] = rfi_filter
    return kwargs


def _resolve_sza_regime(
    *,
    time: dt.datetime | None,
    lat_deg: float | None,
    lon_deg: float | None,
    sza_deg: float | None,
    regime: str | None,
) -> tuple[float | None, str]:
    if (
        sza_deg is None
        and time is not None
        and lat_deg is not None
        and lon_deg is not None
    ):
        sza_deg = solar_zenith_angle(time, lat_deg, lon_deg)
    if regime is None:
        if sza_deg is None:
            raise ValueError(
                "Provide sza_deg, time+lat_deg+lon_deg, or explicit regime."
            )
        regime = classify_solar_regime(float(sza_deg))
    return sza_deg, _valid_regime(regime)


def get_viterbi_preset(
    *,
    time: dt.datetime | None = None,
    lat_deg: float | None = None,
    lon_deg: float | None = None,
    sza_deg: float | None = None,
    regime: str | None = None,
    tuning_preset: str = "balanced",
    e_layer_max_sza: float = 100.0,
    rfi_filter: object | None = None,
) -> ScalerPreset:
    """Return effective Viterbi layer configs and scaler kwargs.

    Args:
        time: Sounding time. Required with ``lat_deg`` and ``lon_deg`` when
            ``sza_deg`` and ``regime`` are not provided.
        lat_deg: Station latitude in degrees north.
        lon_deg: Station longitude in degrees east; west is negative.
        sza_deg: Optional precomputed solar zenith angle.
        regime: Optional explicit ``day``, ``twilight``, or ``night`` regime.
        tuning_preset: One of ``balanced``, ``e_sensitive``, ``f2_smooth``,
            ``strict``, or ``permissive``.
        e_layer_max_sza: Skip normal E-layer scaling at or above this SZA.
        rfi_filter: Optional user-created RFI filter object passed into each
            per-layer scaler kwargs.

    Returns:
        Effective :class:`ScalerPreset` for one ionogram.
    """
    sza_deg, regime = _resolve_sza_regime(
        time=time,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        sza_deg=sza_deg,
        regime=regime,
    )
    tuning_preset = _valid_tuning_preset(tuning_preset)
    regime_params = SOLAR_PARAMETER_SETS[regime]
    tuning = TUNING_PRESETS[tuning_preset]
    e_enabled = sza_deg is None or float(sza_deg) < e_layer_max_sza

    layers = {
        "E": _build_layer("E", regime_params, tuning, sza_deg),
        "F2": _build_layer("F2", regime_params, tuning, sza_deg),
    }
    scaler_kwargs = {
        "E": _build_scaler_kwargs("E", regime_params, tuning, rfi_filter),
        "F2": _build_scaler_kwargs("F2", regime_params, tuning, rfi_filter),
    }
    enabled = {"E": bool(e_enabled), "F2": True}
    metadata = {
        "E_f_hi_effective": layers["E"][0]["f_hi"],
        "F2_f_hi_effective": layers["F2"][0]["f_hi"],
        "e_layer_max_sza": e_layer_max_sza,
        "tuning_description": tuning["description"],
    }
    return ScalerPreset(
        regime=regime,
        sza_deg=sza_deg,
        tuning_preset=tuning_preset,
        layers=layers,
        scaler_kwargs=scaler_kwargs,
        enabled=enabled,
        metadata=metadata,
    )
