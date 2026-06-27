"""Stage 2 trainer for NN-POLAN — station-specific data assimilation.

Takes the Stage 1 foundation checkpoint and fine-tunes it on real ionosonde
virtual-height traces (from RIQ files) using only physics loss.  No IRI
background term — equivalent to 4D-Var "analysis" step where the network
serves as the state-space model.

Loss (Stage 2)
--------------
    L = λ_phy  * L_abel      (Abel integral vs observed h'(f))
      + λ_mono * L_monotone  (unimodal fₚ regulariser)
      [no L_background — we trust the Stage 1 prior implicitly]

Real-data input
---------------
Observed h'(f) traces come from a filtered echo DataFrame produced by
EchoExtractor + IonogramFilter.  For each ionogram, we:
    1. Bin echoes into frequency cells aligned with F_GRID_MHZ
    2. Use median virtual height per cell as the observation
    3. Build an obs_mask marking cells with ≥ min_echoes_per_cell echoes

The conditioning vector c is derived from the station's lat/lon and the
observation timestamp (doy, UT, Kp from a pre-loaded Kp file, F10.7 from NOAA).

Usage
-----
    python trainer_stage2.py \\
        --stage1_ckpt /scratch/$USER/nn_polan/checkpoints/stage1/best.pt \\
        --riq_files   /data/vipir/WI937/*.RIQ \\
        --station_lat 37.9  --station_lon -75.5 \\
        --out_dir     /scratch/$USER/nn_polan/checkpoints/stage2/WI937 \\
        --epochs 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from pynasonde.nn_inversion.adapters.base import resample_trace
from pynasonde.nn_inversion.config import COND_MEAN, COND_STD, NNCfg
from pynasonde.nn_inversion.forward_model import F_GRID_MHZ
from pynasonde.nn_inversion.training.architecture import NNPolan
from pynasonde.nn_inversion.training.physics_loss import PhysicsLoss, abel_invert_batch
from pynasonde.nn_inversion.training.trainer_stage1 import _denorm_ne

_COND_MEAN = COND_MEAN
_COND_STD = COND_STD
_HV_MEAN: float = float(NNCfg.normalisation.hv_mean_km)
_HV_STD: float = float(NNCfg.normalisation.hv_std_km)

_N_F = len(F_GRID_MHZ)


# ---------------------------------------------------------------------------
# Real-trace dataset
# ---------------------------------------------------------------------------


class RealTraceDataset(Dataset):
    """Dataset of real virtual-height traces from ionosonde echoes.

    Parameters
    ----------
    records : list of dicts with keys:
        hv_obs    (N_f,) float32
        obs_mask  (N_f,) bool
        cond      (6,)   float32  [lat, lon, doy, ut, Kp, F10.7]
    """

    def __init__(self, records: list[dict]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int):
        r = self._records[idx]
        cond = r["cond"].astype(np.float32)
        hv = r["hv_obs"].astype(np.float32)
        mask = r["obs_mask"].astype(bool)

        cond_n = torch.tensor((cond - _COND_MEAN) / (_COND_STD + 1e-8))
        hv_n = torch.tensor((hv - _HV_MEAN) / _HV_STD)
        mask_t = torch.tensor(mask)
        hv_km = torch.tensor(hv)  # raw km for physics loss
        return cond_n, hv_n, mask_t, hv_km


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_stage2(args: argparse.Namespace) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if args.verbose else "INFO",
        format="{time:HH:mm:ss} | {level:<8} | {name}: {message}",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}", device)

    # ------------------------------------------------------------------
    # Load real-trace records from pre-saved CSV or Parquet
    # ------------------------------------------------------------------
    records = _load_real_records(args)
    if not records:
        raise ValueError(
            "No valid real traces loaded. Check --echo_file / --riq_files."
        )
    logger.info("Loaded {} real ionogram traces", len(records))

    n_val = max(1, int(0.1 * len(records)))
    n_train = len(records) - n_val
    dataset = RealTraceDataset(records)
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True
    )

    # ------------------------------------------------------------------
    # Model — load Stage 1 weights
    # ------------------------------------------------------------------
    model = NNPolan().to(device)
    ckpt = torch.load(args.stage1_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    logger.info(
        "Loaded Stage 1 checkpoint (epoch {})  val_loss={:.4f}",
        ckpt.get("epoch", "?"),
        ckpt.get("val_loss", float("nan")),
    )

    # ------------------------------------------------------------------
    # Optimiser — lower LR and freeze FiLM generator (keep encoder/decoder)
    # ------------------------------------------------------------------
    # Fine-tune only encoder + decoder; keep FiLM generator frozen to
    # preserve the global conditioning prior from Stage 1.
    frozen = set()
    if args.freeze_film:
        for name, param in model.named_parameters():
            if "film_" in name:
                param.requires_grad_(False)
                frozen.add(name)
        logger.info("Froze {} FiLM parameters.", len(frozen))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Stage 2: no background loss
    criterion = PhysicsLoss(
        lambda_phy=args.lambda_phy,
        lambda_mono=args.lambda_mono,
        lambda_bg=0.0,
    )

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    best_val = float("inf")

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        run = {"total": 0.0, "abel": 0.0, "monotone": 0.0}

        for cond_n, hv_n, mask, hv_km in train_dl:
            cond_n, hv_n, mask, hv_km = (
                x.to(device) for x in (cond_n, hv_n, mask, hv_km)
            )
            ne_abel, btm_mask = abel_invert_batch(hv_km, mask)
            ne_pred_n = model(hv_n, cond_n)
            ne_pred_cm3 = _denorm_ne(ne_pred_n)
            losses = criterion(
                ne_pred=ne_pred_cm3,
                ne_abel_target=ne_abel,
                btm_mask=btm_mask,
            )
            optimiser.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimiser.step()
            for k in run:
                run[k] += losses[k].item()

        scheduler.step()
        for k in run:
            run[k] /= max(len(train_dl), 1)

        model.eval()
        val = {"total": 0.0, "abel": 0.0, "monotone": 0.0}
        with torch.no_grad():
            for cond_n, hv_n, mask, hv_km in val_dl:
                cond_n, hv_n, mask, hv_km = (
                    x.to(device) for x in (cond_n, hv_n, mask, hv_km)
                )
                ne_abel, btm_mask = abel_invert_batch(hv_km, mask)
                ne_pred_cm3 = _denorm_ne(model(hv_n, cond_n))
                vl = criterion(
                    ne_pred=ne_pred_cm3,
                    ne_abel_target=ne_abel,
                    btm_mask=btm_mask,
                )
                for k in val:
                    val[k] += vl[k].item()
        for k in val:
            val[k] /= max(len(val_dl), 1)

        logger.info(
            "Epoch {:3d}/{}  train={:.4f}  val={:.4f}  ({:.0f} s)",
            epoch,
            args.epochs,
            run["total"],
            val["total"],
            time.time() - t0,
        )
        for k in run:
            writer.add_scalar(f"train/{k}", run[k], epoch)
        for k in val:
            writer.add_scalar(f"val/{k}", val[k], epoch)

        is_best = val["total"] < best_val
        if is_best:
            best_val = val["total"]

        ckpt_dict = dict(
            epoch=epoch,
            model=model.state_dict(),
            val_loss=val["total"],
            args=vars(args),
        )
        torch.save(ckpt_dict, out_dir / "last.pt")
        if is_best:
            torch.save(ckpt_dict, out_dir / "best.pt")

    writer.close()
    with open(out_dir / "config.json", "w") as fp:
        json.dump(vars(args), fp, indent=2)
    logger.info("Stage 2 complete. Best val loss: {:.4f}", best_val)


# ---------------------------------------------------------------------------
# Real-trace loader (from pre-processed echo Parquet)
# ---------------------------------------------------------------------------


def _load_real_records(args: argparse.Namespace) -> list[dict]:
    """Load pre-processed echo Parquet and build trace records.

    Expects a Parquet file with columns:
        time (ISO), frequency_khz, height_km, lat, lon, doy, ut_h, kp, f107
    """
    if not args.echo_file:
        logger.warning("--echo_file not provided; returning empty records")
        return []

    df = pd.read_parquet(args.echo_file)

    # Ensure doy / ut_h / kp / f107 columns exist
    if "doy" not in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df["doy"] = df["time"].dt.day_of_year
        df["ut_h"] = df["time"].dt.hour + df["time"].dt.minute / 60.0

    records = []
    for ts, grp in df.groupby("time"):
        freq_mhz = grp["frequency_khz"].to_numpy() / 1e3
        h_km = grp["height_km"].to_numpy()
        hv_obs, obs_mask = resample_trace(freq_mhz, h_km, F_GRID_MHZ)
        # Fill unobserved cells with neutral value (masked out in loss)
        hv_obs = np.where(obs_mask, hv_obs, _HV_MEAN)
        if obs_mask.sum() < 5:  # skip near-empty ionograms
            continue

        kp = float(grp["kp"].iloc[0]) if "kp" in grp.columns else 2.0
        f107 = float(grp["f107"].iloc[0]) if "f107" in grp.columns else 130.0
        lat = float(grp["lat"].iloc[0]) if "lat" in grp.columns else args.station_lat
        lon = float(grp["lon"].iloc[0]) if "lon" in grp.columns else args.station_lon
        doy = float(grp["doy"].iloc[0])
        ut_h = float(grp["ut_h"].iloc[0])

        cond = np.array([lat, lon, doy, ut_h, kp, f107], dtype=np.float32)
        records.append(dict(hv_obs=hv_obs, obs_mask=obs_mask, cond=cond))

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Stage 2: physics-only fine-tuning on real ionosonde data."
    )
    p.add_argument("--stage1_ckpt", required=True, help="Path to Stage 1 best.pt")
    p.add_argument("--echo_file", default=None, help="Parquet file of filtered echoes")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--station_lat", type=float, default=0.0)
    p.add_argument("--station_lon", type=float, default=0.0)
    _t = NNCfg.training
    _m = NNCfg.model
    p.add_argument(
        "--min_valid_pts",
        type=int,
        default=5,
        help="Min F_GRID cells with valid interpolated h' to keep an ionogram",
    )
    p.add_argument("--epochs", type=int, default=int(_t.epochs))
    p.add_argument("--batch", type=int, default=int(_t.batch_size))
    p.add_argument("--lr", type=float, default=float(_t.lr))
    p.add_argument("--wd", type=float, default=float(_t.weight_decay))
    p.add_argument("--lambda_phy", type=float, default=float(_t.lambda_phy))
    p.add_argument("--lambda_mono", type=float, default=float(_t.lambda_mono))
    p.add_argument(
        "--freeze_film",
        action="store_true",
        help="Freeze FiLM generator during Stage 2 fine-tuning",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    train_stage2(args)


if __name__ == "__main__":
    _cli()
