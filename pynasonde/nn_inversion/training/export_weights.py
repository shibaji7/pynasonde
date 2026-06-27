"""Export trained NN-POLAN weights from PyTorch checkpoint → NumPy .npz.

The exported file contains every weight/bias array needed for zero-dependency
inference (pure numpy, no torch at runtime).

Usage
-----
    python export_weights.py \\
        --ckpt  /scratch/$USER/nn_polan/checkpoints/stage2/WI937/best.pt \\
        --out   pynasonde/nn_inversion/weights/WI937.npz

The resulting .npz is loaded by network.py for inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from pynasonde.nn_inversion.config import COND_MEAN, COND_STD, NNCfg
from pynasonde.nn_inversion.forward_model import F_GRID_MHZ, H_GRID_KM
from pynasonde.nn_inversion.training.architecture import NNPolan

_COND_MEAN = COND_MEAN
_COND_STD = COND_STD
_HV_MEAN: float = float(NNCfg.normalisation.hv_mean_km)
_HV_STD: float = float(NNCfg.normalisation.hv_std_km)
_LOG_NE_MIN: float = float(NNCfg.normalisation.log_ne_min)
_LOG_NE_MAX: float = float(NNCfg.normalisation.log_ne_max)
_LATENT_DIM: int = int(NNCfg.model.latent_dim)
_FEAT_DIM: int = int(NNCfg.model.feat_dim)


def export(ckpt_path: str | Path, out_path: str | Path, device: str = "cpu") -> Path:
    """Load checkpoint and write weights to .npz.

    Parameters
    ----------
    ckpt_path : path to .pt checkpoint (Stage 1 or Stage 2)
    out_path  : destination .npz file
    device    : torch device for loading (usually 'cpu' is fine)

    Returns
    -------
    Path to the written .npz file.
    """
    ckpt_path = Path(ckpt_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading checkpoint: {}", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    # Infer model hyper-params from checkpoint args if present
    args = ckpt.get("args", {})
    model = NNPolan(
        latent_dim=args.get("latent_dim", _LATENT_DIM),
        feat_dim=args.get("feat_dim", _FEAT_DIM),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Flatten all parameters to numpy arrays
    weights: dict[str, np.ndarray] = {}
    for name, param in model.state_dict().items():
        weights[name] = param.cpu().numpy()

    # Include normalisation constants so inference code is self-contained
    weights["__cond_mean"] = _COND_MEAN
    weights["__cond_std"] = _COND_STD
    weights["__hv_mean"] = np.array([_HV_MEAN], dtype=np.float32)
    weights["__hv_std"] = np.array([_HV_STD], dtype=np.float32)
    weights["__log_ne_min"] = np.array([_LOG_NE_MIN], dtype=np.float32)
    weights["__log_ne_max"] = np.array([_LOG_NE_MAX], dtype=np.float32)
    weights["__h_grid_km"] = H_GRID_KM.astype(np.float32)
    weights["__f_grid_mhz"] = F_GRID_MHZ.astype(np.float32)
    weights["__epoch"] = np.array([ckpt.get("epoch", -1)])
    weights["__val_loss"] = np.array([ckpt.get("val_loss", float("nan"))])
    weights["__latent_dim"] = np.array([args.get("latent_dim", _LATENT_DIM)])
    weights["__feat_dim"] = np.array([args.get("feat_dim", _FEAT_DIM)])

    np.savez_compressed(out_path, **weights)
    logger.info("Exported {} arrays → {}", len(weights), out_path)
    return out_path


def _cli() -> None:
    p = argparse.ArgumentParser(description="Export NN-POLAN weights to .npz")
    p.add_argument("--ckpt", required=True, help="PyTorch checkpoint (.pt)")
    p.add_argument("--out", required=True, help="Output .npz path")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    out = export(args.ckpt, args.out, args.device)
    print(f"Exported: {out}")


if __name__ == "__main__":
    _cli()
