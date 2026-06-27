"""Independent-loss convergence diagnostic for NN-POLAN.

Answers the question: can each loss term decrease on its own from a given
checkpoint?

Configs
-------
    Config A — BG only    (lambda_phy=0, lambda_mono=0, lambda_bg=1)
    Config B — Abel only  (lambda_phy=1, lambda_mono=0, lambda_bg=0)
    Config C — Mono only  (lambda_phy=0, lambda_mono=1, lambda_bg=0)
    Config D — Abel + BG  (lambda_phy=1, lambda_mono=0, lambda_bg=1)

Each config starts from the SAME checkpoint independently (fresh copy of
weights, fresh optimiser).  The Abel loss uses the inversion path:

    L_abel = mean( (log10 Ne_pred − log10 Ne_abel)²  ×  bottomside_mask )

where Ne_abel is obtained by Abel-inverting the stored h'(f) trace — no
forward Abel model in the computational graph, so gradients are always
well-conditioned.

Interpretation
--------------
B DECREASING ✓  → Abel inversion gradient works; combined training will converge
B FLAT / UP  ✗  → deeper issue (should not occur with inversion loss)
C DECREASING ✓  → Mono gradient fine
D DECREASING ✓  → Abel + BG jointly converge

Usage
-----
    python diagnose_convergence.py \\
        --checkpoint /home/chakras4/nn_polan/checkpoint/best.pt \\
        --data_dir   /home/chakras4/nn_polan \\
        --n_shards   3 \\
        --n_steps    200 \\
        --log_every  20 \\
        --lr         3e-5 \\
        --device     cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))

from pynasonde.nn_inversion.training.architecture import NNPolan
from pynasonde.nn_inversion.training.physics_loss import (
    abel_invert_batch,
    monotone_loss,
)
from pynasonde.nn_inversion.training.trainer_stage1 import ShardDataset, _denorm_ne

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sep(title: str = "") -> None:
    w = 70
    pad = (w - len(title) - 2) // 2
    print("\n" + "─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))


def _load_model(ckpt_path: str, device: torch.device) -> NNPolan:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = NNPolan().to(device)
    model.load_state_dict(ckpt["model"])
    return model


def _run_config(
    config_name: str,
    model: NNPolan,
    loader,
    device: torch.device,
    lr: float,
    n_steps: int,
    log_every: int,
    lambda_phy: float,
    lambda_mono: float,
    lambda_bg: float,
) -> dict[str, list]:
    """Run gradient steps under one loss configuration. Returns history dict."""

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {"step": [], "abel": [], "mono": [], "bg": [], "total": []}

    data_iter = iter(loader)
    step = 0

    while step < n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        cond_n, hv_n, ne_n, ne_raw, hv_km, obs_mask = [x.to(device) for x in batch]

        # Abel inversion target (no forward model in graph — gradient always
        # well-conditioned)
        with torch.no_grad():
            ne_abel, btm_mask = abel_invert_batch(hv_km, obs_mask)

        ne_pred_n = model(hv_n, cond_n)
        ne_pred_cm3 = _denorm_ne(ne_pred_n)

        _zero = torch.zeros(1, device=device)

        if lambda_phy > 0:
            log_pred = torch.log10(ne_pred_cm3.clamp(min=1.0))
            log_abel = torch.log10(ne_abel.to(ne_pred_cm3.dtype).clamp(min=1.0))
            mf = btm_mask.float()
            n_valid = mf.sum().clamp(min=1.0)
            l_abel = ((log_pred - log_abel) ** 2 * mf).sum() / n_valid
        else:
            l_abel = _zero

        l_mono = monotone_loss(ne_pred_cm3) if lambda_mono > 0 else _zero
        l_bg = nn.functional.mse_loss(ne_pred_n, ne_n) if lambda_bg > 0 else _zero

        total = lambda_phy * l_abel + lambda_mono * l_mono + lambda_bg * l_bg

        opt.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        if step % log_every == 0:
            history["step"].append(step)
            history["abel"].append(l_abel.item())
            history["mono"].append(l_mono.item())
            history["bg"].append(l_bg.item())
            history["total"].append(total.item())

            print(
                f"  [{config_name}]  step {step:4d}  "
                f"total={total.item():.3e}  "
                f"abel={l_abel.item():.3e}  "
                f"mono={l_mono.item():.3e}  "
                f"bg={l_bg.item():.3e}"
            )

        step += 1

    return history


def _verdict(history: dict, key: str) -> str:
    """Return DECREASING / FLAT / INCREASING based on first vs last values."""
    vals = history[key]
    if len(vals) < 2:
        return "UNKNOWN"
    first = np.mean(vals[:3])
    last = np.mean(vals[-3:])
    ratio = last / (first + 1e-30)
    if ratio < 0.8:
        return f"DECREASING ✓  ({first:.3e} → {last:.3e}, ratio={ratio:.2f})"
    elif ratio > 1.2:
        return f"INCREASING ✗  ({first:.3e} → {last:.3e}, ratio={ratio:.2f})"
    else:
        return f"FLAT       ~  ({first:.3e} → {last:.3e}, ratio={ratio:.2f})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"\nDevice     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"n_steps    : {args.n_steps}  log_every={args.log_every}")
    print(f"lr         : {args.lr:.2e}")

    shard_paths = sorted(Path(args.data_dir).glob("shard_*.nc"))[: args.n_shards]
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.nc in {args.data_dir}")
    print(f"Shards     : {len(shard_paths)}")

    ds = ShardDataset(shard_paths)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    print(f"Samples    : {len(ds)}")

    configs = [
        # (name,          lambda_phy, lambda_mono, lambda_bg, tracked_key)
        ("A: BG only", 0.0, 0.0, 1.0, "bg"),
        ("B: Abel only", 1.0, 0.0, 0.0, "abel"),
        ("C: Mono only", 0.0, 1.0, 0.0, "mono"),
        ("D: Abel + BG", 1.0, 0.0, 1.0, "abel"),
    ]

    results = {}

    for name, l_phy, l_mono, l_bg, key in configs:
        _sep(name)
        model = _load_model(args.checkpoint, device)
        hist = _run_config(
            config_name=name,
            model=model,
            loader=loader,
            device=device,
            lr=args.lr,
            n_steps=args.n_steps,
            log_every=args.log_every,
            lambda_phy=l_phy,
            lambda_mono=l_mono,
            lambda_bg=l_bg,
        )
        results[name] = (hist, key)

    # ── summary ───────────────────────────────────────────────────────────────
    _sep("SUMMARY")
    print()
    print(f"  {'Config':<16s}  {'Tracked':<8s}  Verdict")
    print(f"  {'──────':<16s}  {'───────':<8s}  ───────")
    for name, (hist, key) in results.items():
        verdict = _verdict(hist, key)
        print(f"  {name:<16s}  {key:<8s}  {verdict}")

    print()
    print("  Interpretation:")
    print("    B DECREASING ✓ → Abel inversion gradient works; training will converge")
    print("    B FLAT / UP  ✗ → unexpected — check checkpoint or data")
    print("    C DECREASING ✓ → Mono gradient fine")
    print("    D DECREASING ✓ → Abel + BG jointly converge")
    _sep()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Diagnose whether each NN-POLAN loss can decrease independently."
    )
    p.add_argument("--checkpoint", required=True, help="Checkpoint .pt to diagnose")
    p.add_argument("--data_dir", required=True, help="Directory with shard_*.nc files")
    p.add_argument("--n_shards", type=int, default=3, help="Number of shards to load")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size")
    p.add_argument("--n_steps", type=int, default=200, help="Gradient steps per config")
    p.add_argument("--log_every", type=int, default=20, help="Print interval (steps)")
    p.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    _cli()
