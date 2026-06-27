"""Physics loss diagnostic script for NN-POLAN.

Runs four unit-test style checks on a saved checkpoint + a handful of data
shards to verify the physics implementation is correct before debugging
training dynamics.

Checks
------
1. monotone_loss(ne_raw)
       Expected : ≈ 0  (IRI profiles are unimodal Chapman-like)
       If large  : Ne unit / normalisation bug

2. abel_loss(ne_raw, torch_forward_batch(ne_raw), obs_mask)
       Expected : exactly 0  (self-consistency of the torch forward model)
       If nonzero: bug inside torch_forward_batch

3. abel_loss(ne_pred, torch_forward_batch(ne_raw), obs_mask)  [warmup-end model]
       Expected : small  (~0.01–0.1, depends on BG floor)
       Tells us : what the Abel loss actually starts at when physics activates

4. monotone_loss(ne_pred)  [warmup-end model]
       Expected : ≈ 0  (model should have learned IRI-like unimodal profiles)
       If large  : model never learned correct Ne shape during warmup

5. Frequency coverage analysis
       For each sample: how many obs_mask freqs have no reflection in ne_pred?
       Tells us : whether the wrong-foF2 hypothesis is correct

Usage
-----
    python diagnose_physics.py \\
        --checkpoint /home/chakras4/nn_polan/checkpoint/epoch_0010.pt \\
        --data_dir   /home/chakras4/nn_polan \\
        --n_shards   3 \\
        --batch_size 256 \\
        --device     cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[7]))  # repo root

from pynasonde.nn_inversion.config import COND_MEAN, COND_STD, NNCfg
from pynasonde.nn_inversion.training.architecture import NNPolan
from pynasonde.nn_inversion.training.physics_loss import (
    PhysicsLoss,
    abel_loss,
    monotone_loss,
    torch_forward_batch,
)
from pynasonde.nn_inversion.training.trainer_stage1 import (
    ShardDataset,
    _denorm_ne,
    _norm_cond,
    _norm_hv,
    _norm_ne,
)

_COND_MEAN = np.array(COND_MEAN, dtype=np.float32)
_COND_STD = np.array(COND_STD, dtype=np.float32)
_HV_MEAN = float(NNCfg.normalisation.hv_mean_km)
_HV_STD = float(NNCfg.normalisation.hv_std_km)


# ── helpers ────────────────────────────────────────────────────────────────────


def _sep(title: str = "") -> None:
    w = 70
    if title:
        pad = (w - len(title) - 2) // 2
        print("\n" + "─" * pad + f" {title} " + "─" * pad)
    else:
        print("\n" + "─" * w)


def _stat(name: str, t: torch.Tensor) -> None:
    v = t.detach().float()
    print(
        f"  {name:<40s}  mean={v.mean():.4e}  std={v.std():.4e}"
        f"  min={v.min():.4e}  max={v.max():.4e}"
    )


# ── main diagnostic ────────────────────────────────────────────────────────────


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"\nDevice : {device}")

    # ── load shards ────────────────────────────────────────────────────────────
    shard_paths = sorted(Path(args.data_dir).glob("shard_*.nc"))[: args.n_shards]
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.nc found in {args.data_dir}")
    print(
        f"Shards : {len(shard_paths)}  ({shard_paths[0].name} … {shard_paths[-1].name})"
    )

    ds = ShardDataset(shard_paths)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    print(f"Samples: {len(ds)}")

    # ── load model ─────────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = NNPolan().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    epoch_ckpt = ckpt.get("epoch", "?")
    print(f"Checkpoint epoch: {epoch_ckpt}  val_loss={ckpt.get('val_loss', '?'):.4e}")

    # ── accumulate results across batches ──────────────────────────────────────
    res = {
        "mono_raw": [],  # check 1
        "abel_self": [],  # check 2: abel(ne_raw, h'_raw)  should be 0
        "abel_pred": [],  # check 3: abel(ne_pred, h'_raw)
        "mono_pred": [],  # check 4
        "no_refl_frac": [],  # check 5: fraction of valid freqs with no pred reflection
        "bg_pred": [],  # MSE(ne_pred_n, ne_n) — replicate training BG loss
        "h_pred_max": [],  # max h' predicted (diagnosis of above-grid integration)
        "h_raw_max": [],  # max h' from IRI
    }

    H_MAX = float(NNCfg.forward_model.h_stop_km)  # 512 km — ceiling of grid

    with torch.no_grad():
        for batch in loader:
            cond_n, hv_n, ne_n, ne_raw, hv_km, obs_mask = [x.to(device) for x in batch]

            # IRI forward (torch) ──────────────────────────────────────────────
            hv_iri = torch_forward_batch(ne_raw.double()).to(ne_raw.dtype)

            # Model prediction ─────────────────────────────────────────────────
            ne_pred_n = model(hv_n, cond_n)
            ne_pred_cm3 = _denorm_ne(ne_pred_n)
            hv_pred = torch_forward_batch(ne_pred_cm3.double()).to(ne_pred_cm3.dtype)

            # ── Check 1: monotone_loss on IRI Ne (should be ≈ 0) ──────────────
            res["mono_raw"].append(monotone_loss(ne_raw).item())

            # ── Check 2: self-consistency — abel(ne_raw, h'_raw) should be 0 ──
            res["abel_self"].append(abel_loss(ne_raw, hv_iri, obs_mask).item())

            # ── Check 3: abel(ne_pred, h'_iri) — real first-epoch Abel loss ───
            res["abel_pred"].append(abel_loss(ne_pred_cm3, hv_iri, obs_mask).item())

            # ── Check 4: monotone_loss on predicted Ne ─────────────────────────
            res["mono_pred"].append(monotone_loss(ne_pred_cm3).item())

            # ── Check 5: fraction of valid freqs where pred has no reflection ──
            # torch_forward_batch returns h' near H_MAX when no reflection occurs.
            # Threshold at 95% of grid ceiling.
            no_refl_thresh = 0.95 * H_MAX  # 486 km
            no_refl_pred = hv_pred > no_refl_thresh  # (B, N_f) bool
            valid_freqs = obs_mask  # (B, N_f) bool
            # Among valid freqs: how many have no reflection in predicted Ne?
            n_valid = valid_freqs.float().sum(dim=1).clamp(min=1)
            n_no_refl = (no_refl_pred & valid_freqs).float().sum(dim=1)
            res["no_refl_frac"].append((n_no_refl / n_valid).mean().item())

            # ── Background loss replicate ──────────────────────────────────────
            res["bg_pred"].append(torch.nn.functional.mse_loss(ne_pred_n, ne_n).item())

            # ── h' range stats ─────────────────────────────────────────────────
            res["h_pred_max"].append(hv_pred[valid_freqs].max().item())
            res["h_raw_max"].append(hv_iri[valid_freqs].max().item())

    # ── print results ──────────────────────────────────────────────────────────
    def avg(key):
        return float(np.mean(res[key]))

    _sep("CHECK 1 — monotone_loss(ne_raw)  [expect ≈ 0]")
    print(f"  mean = {avg('mono_raw'):.4e}")
    print(
        f"  {'PASS ✓' if avg('mono_raw') < 1e-3 else 'FAIL ✗  IRI profiles are not unimodal — Ne unit/normalisation bug?'}"
    )

    _sep("CHECK 2 — abel_loss(ne_raw, torch_fwd(ne_raw))  [expect exactly 0]")
    print(f"  mean = {avg('abel_self'):.4e}")
    print(
        f"  {'PASS ✓' if avg('abel_self') < 1e-6 else 'FAIL ✗  torch_forward_batch is not self-consistent — implementation bug!'}"
    )

    _sep("CHECK 3 — abel_loss(ne_pred, torch_fwd(ne_raw))  [informational]")
    print(f"  mean = {avg('abel_pred'):.4e}")
    print(f"  (this is the Abel loss magnitude at physics-phase start)")
    print(f"  BG loss (replicated) = {avg('bg_pred'):.4e}")

    _sep("CHECK 4 — monotone_loss(ne_pred)  [expect ≈ 0 if model learned IRI shape]")
    print(f"  mean = {avg('mono_pred'):.4e}")
    print(
        f"  {'PASS ✓' if avg('mono_pred') < 1e-2 else 'WARN ⚠  Model Ne is not unimodal — warmup did not converge to IRI shape'}"
    )

    _sep("CHECK 5 — wrong-foF2 hypothesis")
    frac = avg("no_refl_frac")
    print(
        f"  Mean fraction of valid freqs with no reflection in ne_pred : {frac:.3f}  ({frac*100:.1f}%)"
    )
    print(f"  Max h'_pred (valid freqs) : {float(np.mean(res['h_pred_max'])):.1f} km")
    print(f"  Max h'_raw  (valid freqs) : {float(np.mean(res['h_raw_max'])):.1f} km")
    if frac > 0.02:
        print(
            f"  HYPOTHESIS CONFIRMED ✗  {frac*100:.1f}% of valid freqs have no predicted"
        )
        print(
            f"  reflection — these drive Abel loss to ~{frac * (0.95*H_MAX/150)**2:.2f} per sample"
        )
    else:
        print(f"  Hypothesis not supported — wrong-foF2 effect is negligible (<2%)")

    _sep("SUMMARY")
    print(f"  BG floor (this checkpoint) : {avg('bg_pred'):.4e}")
    print(f"  Abel self-consistency      : {avg('abel_self'):.4e}")
    print(f"  Abel at physics start      : {avg('abel_pred'):.4e}")
    print(f"  Monotone (IRI)             : {avg('mono_raw'):.4e}")
    print(f"  Monotone (pred)            : {avg('mono_pred'):.4e}")
    print(f"  No-reflection fraction     : {frac*100:.1f}%")
    _sep()


# ── CLI ────────────────────────────────────────────────────────────────────────


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Diagnose NN-POLAN physics loss implementation."
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a .pt checkpoint (e.g. epoch_0010.pt — end of warmup)",
    )
    p.add_argument(
        "--data_dir",
        required=True,
        help="Directory containing shard_*.nc files",
    )
    p.add_argument(
        "--n_shards",
        type=int,
        default=3,
        help="Number of shard files to load (default 3)",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for evaluation (default 256)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Device: cpu or cuda (default cpu)",
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
