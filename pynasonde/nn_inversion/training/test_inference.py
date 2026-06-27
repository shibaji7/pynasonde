"""Quick inference test for a trained Stage-1 NN-POLAN checkpoint.

Usage
-----
    python test_inference.py \
        --ckpt   /tmp/nn_polan/out/best.pt \
        --data_dir /tmp/nn_polan/ \
        --sample 42 \
        --device cpu

Produces a 3-panel figure:
  Left   — Input ionogram trace h'(f) (IRI-generated)
  Middle — Predicted Ne(h) vs IRI Ne(h) profile
  Right  — Abel(predicted Ne) vs observed h'(f)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from pynasonde.nn_inversion.config import COND_MEAN, COND_STD, NNCfg
from pynasonde.nn_inversion.forward_model import F_GRID_MHZ, H_GRID_KM
from pynasonde.nn_inversion.training.architecture import NNPolan
from pynasonde.nn_inversion.training.physics_loss import torch_forward_batch
from pynasonde.nn_inversion.training.trainer_stage1 import (
    _shard_to_year,
    _split_shards_temporally,
)

# ---------------------------------------------------------------------------
# Normalisation constants — from config.toml [nn_inversion.normalisation]
# ---------------------------------------------------------------------------
_COND_MEAN = COND_MEAN
_COND_STD = COND_STD
_HV_MEAN: float = float(NNCfg.normalisation.hv_mean_km)
_HV_STD: float = float(NNCfg.normalisation.hv_std_km)
_LOG_NE_MIN: float = float(NNCfg.normalisation.log_ne_min)
_LOG_NE_MAX: float = float(NNCfg.normalisation.log_ne_max)


def _norm_cond(c: np.ndarray) -> torch.Tensor:
    return torch.tensor((c - _COND_MEAN) / (_COND_STD + 1e-8), dtype=torch.float32)


def _norm_hv(hv: np.ndarray) -> torch.Tensor:
    return torch.tensor((hv - _HV_MEAN) / _HV_STD, dtype=torch.float32)


def _denorm_ne(ne_norm: torch.Tensor) -> torch.Tensor:
    log_ne = ne_norm * (_LOG_NE_MAX - _LOG_NE_MIN) + _LOG_NE_MIN
    log_ne = log_ne.clamp(_LOG_NE_MIN - 1.0, _LOG_NE_MAX + 2.0)
    return 10.0**log_ne


# ---------------------------------------------------------------------------
# Load one sample from the shard dataset
# ---------------------------------------------------------------------------


def _select_shards(
    data_dir: Path,
    split: str,
    train_end_year: int,
    val_end_year: int,
) -> list[Path]:
    """Return the shard paths for the requested temporal split.

    Parameters
    ----------
    data_dir       : directory containing shard_*.nc files
    split          : one of 'train', 'val', 'test', 'all'
    train_end_year : last year (inclusive) in the training split
    val_end_year   : last year (inclusive) in the validation split
    """
    all_shards = sorted(data_dir.glob("shard_*.nc"))
    if not all_shards:
        raise FileNotFoundError(f"No shard_*.nc files in {data_dir}")

    if split == "all":
        return all_shards

    train_paths, val_paths, test_paths = _split_shards_temporally(
        all_shards, train_end_year, val_end_year
    )
    mapping = {"train": train_paths, "val": val_paths, "test": test_paths}
    paths = mapping[split]
    if not paths:
        raise ValueError(
            f"Split '{split}' produced zero shards. "
            f"Check --train_end_year ({train_end_year}) and --val_end_year ({val_end_year})."
        )
    years = [_shard_to_year(p) for p in paths]
    print(
        f"Split '{split}': {len(paths)} shards  " f"(years {min(years)}–{max(years)})"
    )
    return paths


def load_sample(shards: list[Path], sample_idx: int):
    """Return (ne_cm3, h_virtual, obs_mask, cond) for sample_idx within shards."""
    if not shards:
        raise FileNotFoundError("No shard files provided.")

    offset = 0
    for p in shards:
        ds = xr.open_dataset(p)
        n = ds.dims["sample"]
        if offset + n > sample_idx:
            idx = sample_idx - offset
            ne = ds["ne_cm3"].values[idx].astype(np.float32)
            hv = ds["h_virtual"].values[idx].astype(np.float32)
            mask = ds["obs_mask"].values[idx].astype(bool)
            cond = ds["cond"].values[idx].astype(np.float32)
            ds.close()
            print(f"  Loaded from {p.name}  (local idx={idx}, global idx={sample_idx})")
            return ne, hv, mask, cond
        offset += n
        ds.close()

    raise IndexError(
        f"sample_idx={sample_idx} exceeds split dataset size ({offset} samples)."
    )


# ---------------------------------------------------------------------------
# Main inference + plot
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    # --- load checkpoint ---
    ckpt = torch.load(args.ckpt, map_location=device)
    saved_args = ckpt.get("args", {})
    latent_dim = saved_args.get("latent_dim", args.latent_dim)
    feat_dim = saved_args.get("feat_dim", args.feat_dim)

    model = NNPolan(latent_dim=latent_dim, feat_dim=feat_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint  : {args.ckpt}  (epoch {ckpt.get('epoch','?')})")
    print(f"Val loss at save   : {ckpt.get('val_loss', float('nan')):.4g}")

    # --- select shards for the requested split ---
    shards = _select_shards(
        Path(args.data_dir),
        split=args.split,
        train_end_year=args.train_end_year,
        val_end_year=args.val_end_year,
    )
    print(
        f"Total samples in split: "
        f"{sum(xr.open_dataset(p).dims['sample'] for p in shards)}"
    )

    # --- load sample ---
    for sample_idx in args.sample:
        print(f"\n=== Sample {sample_idx} ===")
        ne_iri, hv_obs, obs_mask, cond = load_sample(shards, sample_idx)
        lat, lon, doy, ut, kp, f107 = cond
        print(
            f"\nSample {sample_idx}: lat={lat:.1f}  lon={lon:.1f}  doy={int(doy)}"
            f"  UT={ut:.0f}h  Kp={kp:.1f}  F10.7={f107:.1f}"
        )

        hv_t = _norm_hv(hv_obs).unsqueeze(0).to(device)
        cond_t = _norm_cond(cond).unsqueeze(0).to(device)

        # --- forward pass ---
        with torch.no_grad():
            ne_pred_n = model(hv_t, cond_t)  # (1, N_h) normalised
            ne_pred_cm3 = _denorm_ne(ne_pred_n)  # (1, N_h) cm⁻³
            hv_pred = torch_forward_batch(ne_pred_cm3.double()).float()  # (1, N_f)

        ne_pred = ne_pred_cm3[0].cpu().numpy()
        hv_pred_np = hv_pred[0].cpu().numpy()

        # mask above-foF2 in predicted trace (NaN where mask=0)
        hv_pred_plot = np.where(obs_mask, hv_pred_np, np.nan)
        hv_obs_plot = np.where(obs_mask, hv_obs, np.nan)

        # --- 3-panel plot ---
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle(
            f"NN-POLAN Stage-1 Inference  |  sample {sample_idx}\n"
            f"lat={lat:.0f}°  lon={lon:.0f}°  DOY={int(doy)}  UT={ut:.0f}h"
            f"  Kp={kp:.1f}  F10.7={f107:.0f}",
            fontsize=10,
        )

        # Panel 1 — input ionogram trace
        ax = axes[0]
        ax.plot(F_GRID_MHZ, hv_obs_plot, "k.", ms=2, label="IRI h'(f) [input]")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Virtual Height (km)")
        ax.set_title("Input Trace h'(f)")
        ax.set_ylim(60, 800)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 2 — Ne profiles
        ax = axes[1]
        ax.semilogx(ne_iri + 1, H_GRID_KM, "b-", lw=1.5, label="IRI Ne (target)")
        ax.semilogx(ne_pred + 1, H_GRID_KM, "r--", lw=1.5, label="NN predicted Ne")
        ax.set_xlabel("Ne (cm⁻³, log scale)")
        ax.set_ylabel("Height (km)")
        ax.set_title("Ne Profile")
        ax.set_ylim(60, 500)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 3 — Abel consistency
        ax = axes[2]
        ax.plot(F_GRID_MHZ, hv_obs_plot, "b.", ms=3, label="Observed h'(f)")
        ax.plot(F_GRID_MHZ, hv_pred_plot, "r--", lw=1, label="Abel(NN Ne)")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Virtual Height (km)")
        ax.set_title("Abel Consistency Check")
        ax.set_ylim(60, 800)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # RMS in km over valid freqs
        valid = obs_mask & np.isfinite(hv_pred_np)
        if valid.any():
            rms = np.sqrt(np.mean((hv_pred_np[valid] - hv_obs[valid]) ** 2))
            axes[2].set_title(f"Abel Consistency  |  RMS={rms:.1f} km")

        plt.tight_layout()
        out_png = (
            Path(args.out)
            if args.out
            else Path(args.ckpt).parent / f"inference_sample{sample_idx}.png"
        )
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"\nSaved → {out_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    p = argparse.ArgumentParser(description="Stage-1 NN-POLAN inference test.")
    p.add_argument("--ckpt", required=True, help="Path to best.pt / last.pt")
    p.add_argument("--data_dir", required=True, help="Dir with shard NetCDF files")
    p.add_argument(
        "--sample",
        type=int,
        nargs="+",
        required=True,
        help=(
            "Sample index (0-based), one or more values. "
            "Single: --sample 5   Range via shell: --sample $(seq 0 9)."
        ),
    )
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--feat_dim", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--out", default=None, help="Output PNG path (optional, single sample only)"
    )
    _t = NNCfg.training
    p.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="test",
        help=(
            "Which temporal split to draw samples from. "
            "'test' = years > val_end_year (held-out), "
            "'val'  = train_end_year < year <= val_end_year, "
            "'train'= year <= train_end_year, "
            "'all'  = all shards (default: test)."
        ),
    )
    p.add_argument(
        "--train_end_year",
        type=int,
        default=int(_t.train_end_year),
        help="Last year (inclusive) in the training split [default from config.toml]",
    )
    p.add_argument(
        "--val_end_year",
        type=int,
        default=int(_t.val_end_year),
        help="Last year (inclusive) in the validation split [default from config.toml]",
    )
    run(p.parse_args())


if __name__ == "__main__":
    _cli()
