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
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from pynasonde.nn_inversion.config import COND_MEAN, COND_STD, NNCfg
from pynasonde.nn_inversion.training.architecture import NNPolan
from pynasonde.nn_inversion.training.physics_loss import PhysicsLoss, abel_invert_batch

# ---------------------------------------------------------------------------
# Normalisation statistics — loaded from config.toml [nn_inversion.normalisation]
# ---------------------------------------------------------------------------
_COND_MEAN = COND_MEAN
_COND_STD = COND_STD
_HV_MEAN: float = float(NNCfg.normalisation.hv_mean_km)
_HV_STD: float = float(NNCfg.normalisation.hv_std_km)
_LOG_NE_MIN: float = float(NNCfg.normalisation.log_ne_min)
_LOG_NE_MAX: float = float(NNCfg.normalisation.log_ne_max)


# ---------------------------------------------------------------------------
# Shard helpers
# ---------------------------------------------------------------------------


def _shard_to_year(shard_path: Path) -> int:
    """Return the year of the first (year, doy) day covered by a shard.

    Reads the ``year_start`` global attribute written by ``synthetic_data.py``.
    Falls back to deriving the year from the shard index when the attribute is
    absent (e.g. shards generated before TODO-1 was applied).

    Parameters
    ----------
    shard_path : Path
        Path to a ``shard_NNNNN.nc`` file.

    Returns
    -------
    int
        Calendar year (e.g. 2017).
    """
    ds = xr.open_dataset(shard_path)
    year = int(ds.attrs.get("year_start", -1))
    if year == -1:
        # Fallback: derive from shard index + config grid
        from pynasonde.nn_inversion.config import NNCfg

        shard_idx = int(ds.attrs.get("shard", 0))
        n_doy = int(NNCfg.data.doy_n)
        year = int(NNCfg.data.year_start) + shard_idx // n_doy
    ds.close()
    return year


def _split_shards_temporally(
    shard_paths: list[Path],
    train_end_year: int,
    val_end_year: int,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Partition shard paths into train / val / test by year.

    Parameters
    ----------
    shard_paths : list[Path]
        All shard files sorted by name.
    train_end_year : int
        Last year (inclusive) belonging to the training split.
    val_end_year : int
        Last year (inclusive) belonging to the validation split.
        Years beyond this go into the test split.

    Returns
    -------
    train_paths, val_paths, test_paths : list[Path]
    """
    train_paths, val_paths, test_paths = [], [], []
    for p in shard_paths:
        yr = _shard_to_year(p)
        if yr <= train_end_year:
            train_paths.append(p)
        elif yr <= val_end_year:
            val_paths.append(p)
        else:
            test_paths.append(p)
    return train_paths, val_paths, test_paths


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
    """Map normalised Ne back to cm⁻³.

    Clamps log_ne to [_LOG_NE_MIN-1, _LOG_NE_MAX+2] (i.e. Ne ∈ [0.1, 3×10⁸] cm⁻³)
    before exponentiation to prevent float32 overflow (>10³⁸) at random init
    when softplus output is large, which would produce inf → NaN in mono_loss
    and inf in background_loss.
    """
    log_ne = ne_norm * (_LOG_NE_MAX - _LOG_NE_MIN) + _LOG_NE_MIN
    log_ne = log_ne.clamp(_LOG_NE_MIN - 1.0, _LOG_NE_MAX + 2.0)
    return 10.0**log_ne


# ---------------------------------------------------------------------------
# Summary plot
# ---------------------------------------------------------------------------


def _plot_training_curves(
    train_hist: dict[str, list[float]],
    val_hist: dict[str, list[float]],
    epoch_nums: list[int],
    test_losses: dict[str, float] | None,
    warmup_epochs: int,
    best_epoch: int,
    out_path: Path,
) -> None:
    """Save a 2×2 figure with per-component loss curves.

    Parameters
    ----------
    train_hist, val_hist : dict
        Per-epoch loss history; keys ``total``, ``abel``, ``monotone``, ``background``.
    epoch_nums : list[int]
        Actual epoch indices corresponding to each history entry (supports resume).
    test_losses : dict or None
        Final held-out test losses.  Plotted as a horizontal dashed line when present.
    warmup_epochs : int
        Epoch index at which physics losses become active; drawn as a vertical line.
    best_epoch : int
        Actual epoch number with the lowest validation total loss.
    out_path : Path
        Destination PNG path.
    """
    _panels = [
        ("total", "Total loss", 0, 0),
        ("abel", "Abel loss", 0, 1),
        ("monotone", "Monotone loss", 1, 0),
        ("background", "Background loss", 1, 1),
    ]

    # layout="constrained" handles suptitle correctly (tight_layout=True is not
    # a valid subplots kwarg and is silently ignored, leaving subplots overlapping)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    fig.suptitle("Stage-1 training curves", fontsize=13)

    for key, title, row, col in _panels:
        ax = axes[row, col]
        ax.plot(epoch_nums, train_hist[key], label="train", color="steelblue")
        ax.plot(epoch_nums, val_hist[key], label="val", color="darkorange")

        # Horizontal test reference line
        if test_losses is not None and key in test_losses:
            ax.axhline(
                test_losses[key],
                color="firebrick",
                linestyle="--",
                linewidth=1.2,
                label=f"test ({test_losses[key]:.4f})",
            )

        # Warmup boundary (only on panels where the transition is visible)
        if warmup_epochs > 0 and key in ("total", "abel", "monotone"):
            ax.axvline(
                warmup_epochs - 0.5,
                color="grey",
                linestyle=":",
                linewidth=0.9,
                label="warmup end",
            )

        # Best-epoch marker — only on total loss panel to avoid clutter
        if epoch_nums and key == "total":
            ax.axvline(
                best_epoch,
                color="green",
                linestyle="-.",
                linewidth=0.9,
                label=f"best (ep {best_epoch})",
            )

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved training curves → {}", out_path)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    """Run Stage 1 training: IRI-supervised + physics-consistent learning.

    Warmup phase (``args.warmup_epochs`` epochs): background loss only (Abel
    and monotone disabled) so the network first learns the IRI prior.
    Physics phase: all three losses enabled according to ``args.lambda_*``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.  See ``_cli()`` for the full argument list.
    """
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
    # Data — temporal split
    # ------------------------------------------------------------------
    shard_paths = sorted(Path(args.data_dir).glob("shard_*.nc"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard NetCDF files found in {args.data_dir}")
    logger.info("Found {} shard files", len(shard_paths))

    if args.max_shards > 0 and len(shard_paths) > args.max_shards:
        # Evenly-spaced sampling across the full sorted list so all years
        # are represented and the temporal split produces non-empty splits.
        indices = np.linspace(0, len(shard_paths) - 1, args.max_shards, dtype=int)
        shard_paths = [shard_paths[i] for i in indices]
        logger.info(
            "Downsampled to {} evenly-spaced shards (--max_shards)", args.max_shards
        )

    train_paths, val_paths, test_paths = _split_shards_temporally(
        shard_paths,
        train_end_year=args.train_end_year,
        val_end_year=args.val_end_year,
    )
    if not train_paths:
        raise ValueError(
            "Temporal split produced zero training shards — check --train_end_year"
        )
    if not val_paths:
        raise ValueError(
            "Temporal split produced zero validation shards — check --val_end_year"
        )
    if not test_paths:
        logger.warning(
            "Temporal split produced zero test shards — test loss will be skipped"
        )

    train_ds = ShardDataset(train_paths)
    val_ds = ShardDataset(val_paths)
    test_ds = ShardDataset(test_paths) if test_paths else None

    logger.info(
        "Temporal split  train: {} samples (≤{})  val: {} samples ({}-{})  "
        "test: {} samples (>{})  ",
        len(train_ds),
        args.train_end_year,
        len(val_ds),
        args.train_end_year + 1,
        args.val_end_year,
        len(test_ds) if test_ds else 0,
        args.val_end_year,
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
    test_dl = (
        DataLoader(
            test_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        if test_ds
        else None
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = NNPolan(
        latent_dim=args.latent_dim,
        feat_dim=args.feat_dim,
    ).to(device)
    logger.info("Model parameters: {}", model.n_params())

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        logger.info("Resumed from {} (epoch {})", args.resume, ckpt.get("epoch", "?"))

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
    best_epoch = 0
    start_epoch = 0

    # Per-epoch history: dict[loss_key → list[float per epoch]]
    _loss_keys = ("total", "abel", "monotone", "background")
    train_hist: dict[str, list[float]] = {k: [] for k in _loss_keys}
    val_hist: dict[str, list[float]] = {k: [] for k in _loss_keys}
    epoch_nums: list[int] = []  # actual epoch numbers (supports resume)

    if args.resume:
        start_epoch = ckpt.get("epoch", 0) + 1

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        running = {"total": 0.0, "abel": 0.0, "monotone": 0.0, "background": 0.0}

        # Curriculum: during warmup epochs disable Abel so the network first
        # learns the Ne shape from the supervised background loss, then
        # transitions to physics-informed training once Ne is non-trivial.
        in_warmup = epoch < args.warmup_epochs
        if in_warmup:
            criterion.lambda_phy = 0.0
            criterion.lambda_mono = 0.0
            criterion.lambda_bg = 1.0
        else:
            # Linearly ramp lambda_phy from 0 → args.lambda_phy over ramp_epochs
            # physics epochs.  At the first physics epoch the Abel loss is
            # typically ~19000× larger than the BG loss; hitting it at full
            # strength collapses Ne.  The ramp lets the model adapt gradually.
            physics_epoch = epoch - args.warmup_epochs  # 0, 1, 2, …
            ramp_frac = min(
                1.0, (physics_epoch + 1) / max(args.lambda_phy_ramp_epochs, 1)
            )
            criterion.lambda_phy = ramp_frac * args.lambda_phy
            criterion.lambda_mono = args.lambda_mono
            criterion.lambda_bg = args.lambda_bg
        if in_warmup and epoch == 0:
            logger.info(
                "Warmup phase: {} epochs bg-only (Abel disabled)", args.warmup_epochs
            )
        if not in_warmup and epoch == args.warmup_epochs:
            logger.info(
                "Warmup complete — ramping Abel loss over {} physics epochs "
                "(λ_phy 0 → {:.3e})",
                args.lambda_phy_ramp_epochs,
                args.lambda_phy,
            )

        for step, batch in enumerate(train_dl):
            cond_n, hv_n, ne_n, ne_raw, hv_km, obs_mask = [x.to(device) for x in batch]

            # Abel inversion: h'_obs(f) → Ne_abel(h) target on H_GRID_KM.
            # Replaces the forward Abel model in the loss — gradient of
            # MSE(Ne_pred, Ne_abel) is always well-conditioned (no singularity).
            with torch.no_grad():
                ne_abel, btm_mask = abel_invert_batch(hv_km, obs_mask)

            # Network prediction (in normalised Ne space)
            ne_pred_n = model(hv_n, cond_n)  # (B, N_h) normalised

            # Convert to physical units for physics loss
            ne_pred_cm3 = _denorm_ne(ne_pred_n)  # (B, N_h)

            losses = criterion(
                ne_pred=ne_pred_cm3,
                ne_iri=ne_n,  # normalised log Ne — keeps bg loss O(1)
                ne_pred_n=ne_pred_n,  # normalised prediction for bg loss
                ne_abel_target=ne_abel,  # Abel inversion target [cm⁻³]
                btm_mask=btm_mask,  # valid bottomside heights
            )

            optimiser.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            for k, v in losses.items():
                running[k] += v.item()

            if step % 100 == 0:
                logger.debug(
                    "  epoch {}  step {}/{}  loss={:.3e}  abel={:.3e}  mono={:.3e}  bg={:.3e}",
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
                ne_abel, btm_mask = abel_invert_batch(hv_km, obs_mask)
                ne_pred_n = model(hv_n, cond_n)
                ne_pred_cm3 = _denorm_ne(ne_pred_n)
                vl = criterion(
                    ne_pred=ne_pred_cm3,
                    ne_iri=ne_n,
                    ne_pred_n=ne_pred_n,
                    ne_abel_target=ne_abel,
                    btm_mask=btm_mask,
                )
                for k in val_losses:
                    val_losses[k] += vl[k].item()
        for k in val_losses:
            val_losses[k] /= max(len(val_dl), 1)

        # Accumulate per-epoch history (TODO-7)
        epoch_nums.append(epoch)
        for k in _loss_keys:
            train_hist[k].append(running[k])
            val_hist[k].append(val_losses[k])

        elapsed = time.time() - t0
        logger.info(
            "Epoch {:3d}/{}  ({:.0f} s)  lr={:.2e}  λ_phy={:.3f}\n"
            "  train  total={:.3e}  abel={:.3e}  bg={:.3e}  mono={:.3e}\n"
            "  val    total={:.3e}  abel={:.3e}  bg={:.3e}  mono={:.3e}",
            epoch,
            args.epochs,
            elapsed,
            optimiser.param_groups[0]["lr"],
            criterion.lambda_phy,
            running["total"],
            running["abel"],
            running["background"],
            running["monotone"],
            val_losses["total"],
            val_losses["abel"],
            val_losses["background"],
            val_losses["monotone"],
        )

        # TensorBoard — train & val
        for k in running:
            writer.add_scalar(f"train/{k}", running[k], epoch)
        for k in val_losses:
            writer.add_scalar(f"val/{k}", val_losses[k], epoch)
        writer.add_scalar("lr", optimiser.param_groups[0]["lr"], epoch)

        # Checkpoint
        is_best = val_losses["total"] < best_val_loss
        if is_best:
            best_val_loss = val_losses["total"]
            best_epoch = epoch

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
            logger.info(
                "  → new best  val total={:.3e}  abel={:.3e}  bg={:.3e}",
                best_val_loss,
                val_losses["abel"],
                val_losses["background"],
            )
        if epoch % args.save_every == 0:
            torch.save(ckpt_dict, out_dir / f"epoch_{epoch:04d}.pt")

        # Per-epoch CSV log (TODO-12): append one row per epoch
        csv_path = out_dir / "training_log.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as csvf:
            writer_csv = csv.writer(csvf)
            if write_header:
                writer_csv.writerow(
                    ["epoch"]
                    + [f"train_{k}" for k in _loss_keys]
                    + [f"val_{k}" for k in _loss_keys]
                    + ["lr"]
                )
            writer_csv.writerow(
                [epoch]
                + [running[k] for k in _loss_keys]
                + [val_losses[k] for k in _loss_keys]
                + [optimiser.param_groups[0]["lr"]]
            )

    # ------------------------------------------------------------------
    # Post-training test evaluation (TODO-8)
    # ------------------------------------------------------------------
    test_losses: dict[str, float] | None = None
    if test_dl is not None:
        logger.info("Running test evaluation …")
        model.eval()
        test_acc = {"total": 0.0, "abel": 0.0, "monotone": 0.0, "background": 0.0}
        with torch.no_grad():
            for batch in test_dl:
                cond_n, hv_n, ne_n, ne_raw, hv_km, obs_mask = [
                    x.to(device) for x in batch
                ]
                ne_abel, btm_mask = abel_invert_batch(hv_km, obs_mask)
                ne_pred_n = model(hv_n, cond_n)
                ne_pred_cm3 = _denorm_ne(ne_pred_n)
                tl = criterion(
                    ne_pred=ne_pred_cm3,
                    ne_iri=ne_n,
                    ne_pred_n=ne_pred_n,
                    ne_abel_target=ne_abel,
                    btm_mask=btm_mask,
                )
                for k in test_acc:
                    test_acc[k] += tl[k].item()
        for k in test_acc:
            test_acc[k] /= max(len(test_dl), 1)
        test_losses = test_acc
        logger.info(
            "Test  total={:.4f}  abel={:.4f}  mono={:.4f}  bg={:.4f}",
            test_losses["total"],
            test_losses["abel"],
            test_losses["monotone"],
            test_losses["background"],
        )
        # TensorBoard test scalars (TODO-13)
        for k, v in test_losses.items():
            writer.add_scalar(f"test/{k}", v, args.epochs - 1)
    else:
        logger.warning("No test shards — skipping test evaluation")

    writer.close()

    # Update best checkpoint with test loss (TODO-9)
    if test_losses is not None:
        best_ckpt_path = out_dir / "best.pt"
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
            best_ckpt["test_loss"] = test_losses["total"]
            torch.save(best_ckpt, best_ckpt_path)

    # Save training config with test loss (TODO-9)
    cfg_dict = vars(args)
    if test_losses is not None:
        cfg_dict["test_loss"] = test_losses["total"]
    with open(out_dir / "config.json", "w") as fp:
        json.dump(cfg_dict, fp, indent=2)

    # Summary plots (TODO-10 / TODO-11)
    _plot_training_curves(
        train_hist=train_hist,
        val_hist=val_hist,
        epoch_nums=epoch_nums,
        test_losses=test_losses,
        warmup_epochs=args.warmup_epochs,
        best_epoch=best_epoch,
        out_path=out_dir / "training_curves.png",
    )

    logger.info(
        "Training complete. Best val loss: {:.4f} (epoch {})", best_val_loss, best_epoch
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Stage 1: train global NN-POLAN foundation model on IRI data."
    )
    p.add_argument("--data_dir", required=True, help="Dir with shard NetCDF files")
    p.add_argument("--out_dir", required=True, help="Checkpoint output dir")
    _t = NNCfg.training
    _m = NNCfg.model
    p.add_argument("--resume", default=None, help="Path to checkpoint .pt to resume")
    p.add_argument("--epochs", type=int, default=int(_t.epochs))
    p.add_argument("--batch", type=int, default=int(_t.batch_size), help="Batch size")
    p.add_argument("--lr", type=float, default=float(_t.lr))
    p.add_argument(
        "--wd", type=float, default=float(_t.weight_decay), help="AdamW weight decay"
    )
    p.add_argument(
        "--workers", type=int, default=int(_t.workers), help="DataLoader workers"
    )
    p.add_argument("--latent_dim", type=int, default=int(_m.latent_dim))
    p.add_argument("--feat_dim", type=int, default=int(_m.feat_dim))
    p.add_argument("--lambda_phy", type=float, default=float(_t.lambda_phy))
    p.add_argument("--lambda_mono", type=float, default=float(_t.lambda_mono))
    p.add_argument("--lambda_bg", type=float, default=float(_t.lambda_bg))
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=int(_t.warmup_epochs),
        help=(
            "Epochs with bg loss only (Abel disabled). "
            "Prevents Ne collapse before network learns Ne shape. "
            "Set 0 to skip. [config: nn_inversion.training.warmup_epochs]"
        ),
    )
    p.add_argument(
        "--lambda_phy_ramp_epochs",
        type=int,
        default=int(_t.lambda_phy_ramp_epochs),
        help=(
            "Number of physics epochs over which lambda_phy is linearly ramped "
            "from 0 to lambda_phy.  Prevents Abel-gradient collapse at the first "
            "physics epoch when the Abel loss is orders of magnitude larger than "
            "the background loss. [config: nn_inversion.training.lambda_phy_ramp_epochs]"
        ),
    )
    p.add_argument(
        "--save_every",
        type=int,
        default=int(_t.save_every),
        help="Extra checkpoint interval",
    )
    p.add_argument(
        "--train_end_year",
        type=int,
        default=int(_t.train_end_year),
        help="Last year (inclusive) in training split [config: nn_inversion.training.train_end_year]",
    )
    p.add_argument(
        "--val_end_year",
        type=int,
        default=int(_t.val_end_year),
        help="Last year (inclusive) in validation split; years beyond go to test "
        "[config: nn_inversion.training.val_end_year]",
    )
    p.add_argument(
        "--max_shards",
        type=int,
        default=0,
        help="Limit total shards loaded via evenly-spaced sampling before the "
        "temporal split, so train/val/test all shrink proportionally. "
        "0 = use all shards (default).",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    _cli()
