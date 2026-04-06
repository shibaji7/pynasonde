"""Stage 1 trainer for NN-POLAN.

Global foundation model trained on IRI-generated synthetic data.

Loss
----
    L = λ_phy  * L_abel          (Abel integral consistency)
      + λ_mono * L_monotone      (unimodal fₚ profile regulariser)
      + λ_bg   * L_background    (MSE vs IRI — 4D-Var background cost)

Input NetCDF shards are produced by synthetic_data.py.  Each shard file contains:
    ne_cm3    (sample, height_km)  float32
    h_virtual (sample, freq_mhz)   float32  — 0-filled above foF2
    obs_mask  (sample, freq_mhz)   int8     — 1 where freq < foF2
    cond      (sample, cond_dim)   float32  [lat, lon, doy, ut, Kp, F10.7]

Usage
-----
Single-GPU run::

    python trainer_stage1.py \\
        --data_dir /scratch/$USER/nn_polan/synthetic \\
        --out_dir  /scratch/$USER/nn_polan/checkpoints/stage1 \\
        --epochs 50 --batch 256

VEGA multi-node (via slurm/train_stage1.sh)::

    sbatch slurm/train_stage1.sh
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from pynasonde.vipir.analysis.nn_inversion.training.architecture import NNPolan
from pynasonde.vipir.analysis.nn_inversion.training.physics_loss import PhysicsLoss

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditioning-vector normalisation statistics
# (computed from grid extents — no data-dependent fitting needed)
# ---------------------------------------------------------------------------
#  [lat, lon, doy, ut_h, kp, f107]
_COND_MEAN = np.array([0.0, 0.0, 183.0, 9.0, 3.5, 135.0], dtype=np.float32)
_COND_STD = np.array([52.0, 104.0, 105.0, 6.3, 2.1, 47.0], dtype=np.float32)

# Virtual-height normalisation (physical bounds)
_HV_MEAN = 300.0  # km
_HV_STD = 150.0  # km

# Electron density normalisation (log-scale is more natural for Ne)
_LOG_NE_MIN = -2.0  # log10(cm⁻³) — effectively 0 for background
_LOG_NE_MAX = 6.5  # log10(cm⁻³) — upper limit of IRI


# ---------------------------------------------------------------------------
# NetCDF shard dataset
# ---------------------------------------------------------------------------


class ShardDataset(Dataset):
    """Loads training samples across all NetCDF shard files produced by synthetic_data.py.

    Each shard is opened once and held in memory as numpy arrays — NetCDF files
    at this size (~10 k samples × 904 heights) fit easily in RAM per shard.
    Samples are concatenated across shards at construction time.
    """

    def __init__(self, shard_paths: list[Path]) -> None:
        ne_list: list[np.ndarray] = []
        hv_list: list[np.ndarray] = []
        mask_list: list[np.ndarray] = []
        cond_list: list[np.ndarray] = []

        for p in shard_paths:
            ds = xr.open_dataset(p)
            ne_list.append(ds["ne_cm3"].values.astype(np.float32))
            hv_list.append(ds["h_virtual"].values.astype(np.float32))
            mask_list.append(ds["obs_mask"].values.astype(np.float32))
            cond_list.append(ds["cond"].values.astype(np.float32))
            ds.close()

        self._ne = np.concatenate(ne_list, axis=0)
        self._hv = np.concatenate(hv_list, axis=0)
        self._mask = np.concatenate(mask_list, axis=0)
        self._cond = np.concatenate(cond_list, axis=0)

    def __len__(self) -> int:
        return len(self._ne)

    def __getitem__(self, idx: int):
        ne = self._ne[idx]
        hv = self._hv[idx]
        mask = self._mask[idx]
        cond = self._cond[idx]
        return (
            _norm_cond(cond),  # normalised conditioning vector
            _norm_hv(hv),  # normalised virtual-height trace
            _norm_ne(ne),  # normalised log Ne (not used in loss, kept for future)
            torch.tensor(ne),  # raw Ne [cm⁻³] for background loss
            torch.tensor(hv),  # raw h_virtual [km] for physics loss
            torch.tensor(mask).bool(),  # obs_mask: True where freq < foF2
        )

    def close(self) -> None:
        pass  # arrays are in memory; nothing to close


def _norm_cond(c: np.ndarray) -> torch.Tensor:
    return torch.tensor((c - _COND_MEAN) / (_COND_STD + 1e-8))


def _norm_hv(hv: np.ndarray) -> torch.Tensor:
    return torch.tensor((hv - _HV_MEAN) / _HV_STD)


def _norm_ne(ne: np.ndarray) -> torch.Tensor:
    log_ne = np.log10(np.maximum(ne, 1e-2))
    return torch.tensor((log_ne - _LOG_NE_MIN) / (_LOG_NE_MAX - _LOG_NE_MIN))


def _denorm_ne(ne_norm: torch.Tensor) -> torch.Tensor:
    """Map normalised Ne back to cm⁻³."""
    log_ne = ne_norm * (_LOG_NE_MAX - _LOG_NE_MIN) + _LOG_NE_MIN
    return 10.0**log_ne


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    shard_paths = sorted(Path(args.data_dir).glob("shard_*.nc"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard NetCDF files found in {args.data_dir}")
    logger.info("Found %d shard files", len(shard_paths))

    dataset = ShardDataset(shard_paths)
    n_val = max(1, int(0.05 * len(dataset)))  # 5 % validation
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    logger.info("Train samples: %d   Val samples: %d", n_train, n_val)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = NNPolan(
        latent_dim=args.latent_dim,
        feat_dim=args.feat_dim,
    ).to(device)
    logger.info("Model parameters: %d", model.n_params())

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        logger.info("Resumed from %s (epoch %d)", args.resume, ckpt.get("epoch", "?"))

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    criterion = PhysicsLoss(
        lambda_phy=args.lambda_phy,
        lambda_mono=args.lambda_mono,
        lambda_bg=args.lambda_bg,
    )

    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    # ------------------------------------------------------------------
    # Epoch loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    start_epoch = 0

    if args.resume:
        start_epoch = ckpt.get("epoch", 0) + 1

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        running = {"total": 0.0, "abel": 0.0, "monotone": 0.0, "background": 0.0}

        for step, batch in enumerate(train_dl):
            cond_n, hv_n, ne_n, ne_raw, hv_km, obs_mask = [x.to(device) for x in batch]

            # Network prediction (in normalised Ne space)
            ne_pred_n = model(hv_n, cond_n)  # (B, N_h) normalised

            # Convert to physical units for physics loss
            ne_pred_cm3 = _denorm_ne(ne_pred_n)  # (B, N_h)

            losses = criterion(
                ne_pred=ne_pred_cm3,
                h_virt_obs=hv_km,  # raw km — not re-derived from hv_n
                ne_iri=ne_raw,
                obs_mask=obs_mask,  # mask above-foF2 freqs from Abel loss
            )

            optimiser.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            for k, v in losses.items():
                running[k] += v.item()

            if step % 100 == 0:
                logger.debug(
                    "  epoch %d  step %d/%d  loss=%.4f  abel=%.4f  mono=%.4f  bg=%.4f",
                    epoch,
                    step,
                    len(train_dl),
                    losses["total"].item(),
                    losses["abel"].item(),
                    losses["monotone"].item(),
                    losses["background"].item(),
                )

        scheduler.step()

        # Normalise by steps
        n_steps = len(train_dl)
        for k in running:
            running[k] /= n_steps

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        model.eval()
        val_losses = {"total": 0.0, "abel": 0.0, "monotone": 0.0, "background": 0.0}
        with torch.no_grad():
            for batch in val_dl:
                cond_n, hv_n, ne_n, ne_raw, hv_km, obs_mask = [
                    x.to(device) for x in batch
                ]
                ne_pred_n = model(hv_n, cond_n)
                ne_pred_cm3 = _denorm_ne(ne_pred_n)
                vl = criterion(
                    ne_pred=ne_pred_cm3,
                    h_virt_obs=hv_km,
                    ne_iri=ne_raw,
                    obs_mask=obs_mask,
                )
                for k in val_losses:
                    val_losses[k] += vl[k].item()
        for k in val_losses:
            val_losses[k] /= max(len(val_dl), 1)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d  train=%.4f  val=%.4f  (%.0f s)  lr=%.2e",
            epoch,
            args.epochs,
            running["total"],
            val_losses["total"],
            elapsed,
            optimiser.param_groups[0]["lr"],
        )

        # TensorBoard
        for k in running:
            writer.add_scalar(f"train/{k}", running[k], epoch)
        for k in val_losses:
            writer.add_scalar(f"val/{k}", val_losses[k], epoch)
        writer.add_scalar("lr", optimiser.param_groups[0]["lr"], epoch)

        # Checkpoint
        is_best = val_losses["total"] < best_val_loss
        if is_best:
            best_val_loss = val_losses["total"]

        ckpt_dict = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_loss": val_losses["total"],
            "args": vars(args),
        }
        torch.save(ckpt_dict, out_dir / "last.pt")
        if is_best:
            torch.save(ckpt_dict, out_dir / "best.pt")
            logger.info("  → new best val loss: %.4f", best_val_loss)
        if epoch % args.save_every == 0:
            torch.save(ckpt_dict, out_dir / f"epoch_{epoch:04d}.pt")

    writer.close()
    dataset.close()

    # Save training config
    with open(out_dir / "config.json", "w") as fp:
        json.dump(vars(args), fp, indent=2)
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Stage 1: train global NN-POLAN foundation model on IRI data."
    )
    p.add_argument("--data_dir", required=True, help="Dir with shard NetCDF files")
    p.add_argument("--out_dir", required=True, help="Checkpoint output dir")
    p.add_argument("--resume", default=None, help="Path to checkpoint .pt to resume")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--feat_dim", type=int, default=128)
    p.add_argument("--lambda_phy", type=float, default=1.0)
    p.add_argument("--lambda_mono", type=float, default=0.01)
    p.add_argument("--lambda_bg", type=float, default=0.1)
    p.add_argument(
        "--save_every", type=int, default=10, help="Extra checkpoint interval"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    _cli()
