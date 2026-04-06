"""Physics loss terms for NN-POLAN training.

All losses operate on PyTorch tensors and are differentiable.

Loss terms
----------
abel_loss
    Core physics constraint: forward-model consistency.
    L_phy = MSE( h'_pred(f),  h'_obs(f) )
    where h'_pred is computed by running the Abel integral (forward_batch)
    on the network's predicted Ne profile.

    Because forward_batch is written in NumPy, we implement a differentiable
    PyTorch version of the Abel integral here (torch_forward_batch).

monotone_loss
    Encourages the plasma-frequency profile fₚ(h) to be unimodal
    (single-humped) — a physically motivated regulariser that penalises
    a second peak above hmF2.

kl_loss
    Background cost term analogous to J_b in 4D-Var:
        L_bg = KL( q(Ne) ‖ p_IRI(Ne) )
    Used only in Stage 1 (supervised) to keep the network close to IRI.
    Implemented as a simple MSE vs. the IRI-generated target profile.

Usage
-----
    from pynasonde.vipir.analysis.nn_inversion.training.physics_loss import (
        PhysicsLoss, torch_forward_batch
    )
    criterion = PhysicsLoss(lambda_phy=1.0, lambda_mono=0.01, lambda_bg=0.1)
    loss = criterion(ne_pred, h_virt_obs, ne_iri=ne_iri_batch)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pynasonde.vipir.analysis.nn_inversion.forward_model import (
    _FP2_CONST,
    _REFLECT_GUARD,
    F_GRID_MHZ,
    H_GRID_KM,
)

# ---------------------------------------------------------------------------
# Constants (pre-compute and register as buffers in the loss module)
# ---------------------------------------------------------------------------
_N_H = len(H_GRID_KM)
_N_F = len(F_GRID_MHZ)

# Height spacing (uniform grid assumed)
_DH_KM = float(H_GRID_KM[1] - H_GRID_KM[0])  # 0.5 km
_H_BASE = float(H_GRID_KM[0])  # 60.0 km

# Pre-compute static tensors on CPU; moved to device in forward
_H_TENSOR = torch.tensor(H_GRID_KM, dtype=torch.float64)  # (N_h,)
_F_TENSOR = torch.tensor(F_GRID_MHZ, dtype=torch.float64)  # (N_f,)


# ---------------------------------------------------------------------------
# Differentiable forward model (Abel integral in PyTorch)
# ---------------------------------------------------------------------------


def torch_forward_batch(
    ne_cm3: torch.Tensor,
    h_grid: torch.Tensor | None = None,
    f_grid: torch.Tensor | None = None,
    reflect_guard: float = _REFLECT_GUARD,
) -> torch.Tensor:
    """Differentiable Abel integral: Ne(h) → h'(f).

    Implements the same physics as forward_model.forward_batch but in PyTorch
    so gradients flow back to the network weights.

    Parameters
    ----------
    ne_cm3 : (B, N_h) electron density [cm⁻³], must be ≥ 0
    h_grid : (N_h,) height grid [km]  — defaults to H_GRID_KM
    f_grid : (N_f,) frequency grid [MHz] — defaults to F_GRID_MHZ
    reflect_guard : fraction of f² below which we treat the layer as
                    reflecting (stops the integral)

    Returns
    -------
    h_virtual : (B, N_f) virtual heights [km]
    """
    device = ne_cm3.device
    dtype = ne_cm3.dtype

    if h_grid is None:
        h_grid = _H_TENSOR.to(device=device, dtype=dtype)
    if f_grid is None:
        f_grid = _F_TENSOR.to(device=device, dtype=dtype)

    B = ne_cm3.shape[0]
    n_h = h_grid.shape[0]
    n_f = f_grid.shape[0]
    dh = h_grid[1] - h_grid[0]
    h_base = h_grid[0]

    # Plasma frequency squared: fₚ²(h) = Ne / _FP2_CONST [MHz²]
    # (same relation as ne_to_fp: fₚ = sqrt(Ne / _FP2_CONST))
    fp2 = ne_cm3 / _FP2_CONST  # (B, N_h)

    # Broadcast for all frequencies: ratio = fₚ²(h) / f²
    #   fp2  : (B, N_h, 1)
    #   f2   : (1,  1,  N_f)
    fp2_b = fp2.unsqueeze(2)  # (B, N_h, 1)
    f2 = (f_grid**2).reshape(1, 1, n_f)  # (1, 1, N_f)
    ratio = fp2_b / (f2 + 1e-30)  # (B, N_h, N_f)

    # group refractive index μ' = 1/√(1 − ratio), clamped below reflection
    ratio_safe = torch.clamp(ratio, max=reflect_guard)
    mu = 1.0 / torch.sqrt(1.0 - ratio_safe + 1e-30)  # (B, N_h, N_f)

    # Reflection mask: find first height where ratio >= guard per (batch, freq)
    above_guard = ratio >= reflect_guard  # (B, N_h, N_f) bool

    # cumulative OR from bottom to top — once True, stays True
    reflect_mask = torch.cummax(above_guard.float(), dim=1).values  # (B, N_h, N_f)
    # reflect_mask[b, k, f] = 1.0 if height k is at or above reflection for freq f

    # Zero out μ' above reflection
    mu = mu * (1.0 - reflect_mask)  # (B, N_h, N_f)

    # Abel integral via trapezoidal rule along height axis
    # h_virtual[b, f] = h_base + Σ_k  mu[b, k, f] * dh
    # (trapezoidal: average adjacent μ' values)
    mu_mid = 0.5 * (mu[:, :-1, :] + mu[:, 1:, :])  # (B, N_h-1, N_f)
    h_virtual = h_base + dh * mu_mid.sum(dim=1)  # (B, N_f)

    return h_virtual


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------


def abel_loss(
    ne_pred: torch.Tensor,
    h_virt_obs: torch.Tensor,
    obs_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Physics consistency loss: Abel(Ne_pred) vs observed h'(f).

    Parameters
    ----------
    ne_pred    : (B, N_h) predicted electron density [cm⁻³]
    h_virt_obs : (B, N_f) observed (or IRI-generated) virtual heights [km]
    obs_mask   : (B, N_f) bool — True where h_virt_obs is valid
                 (e.g., False for frequencies above foF2)

    Returns
    -------
    Scalar loss value.
    """
    h_virt_pred = torch_forward_batch(ne_pred.double()).to(ne_pred.dtype)  # (B, N_f)
    diff = h_virt_pred - h_virt_obs
    if obs_mask is not None:
        diff = diff * obs_mask.float()
        n_valid = obs_mask.float().sum().clamp(min=1.0)
        return (diff**2).sum() / n_valid
    return (diff**2).mean()


def monotone_loss(ne_pred: torch.Tensor) -> torch.Tensor:
    """Penalise secondary peaks in the plasma-frequency profile above hmF2.

    Encourages fₚ(h) to be unimodal (single Chapman-like hump).

    L_mono = mean( ReLU(-Δfₚ_ascending + ε) )  for h < h_peak
           + mean( ReLU(+Δfₚ_descending + ε) )  for h > h_peak

    where ε is a small tolerance to allow grid-scale non-monotonicity.

    Parameters
    ----------
    ne_pred : (B, N_h) — we compute fₚ² ∝ Ne

    Returns
    -------
    Scalar loss.
    """
    fp2 = ne_pred / _FP2_CONST  # (B, N_h)  fₚ²[MHz²]
    diff = fp2[:, 1:] - fp2[:, :-1]  # (B, N_h-1) — positive = ascending

    # Find approximate peak index per profile
    peak_idx = fp2.argmax(dim=1)  # (B,)

    # Build mask: ascending region is 0..peak-1, descending is peak..N_h-2
    n_h = fp2.shape[1]
    idx = torch.arange(n_h - 1, device=ne_pred.device).unsqueeze(0)  # (1, N_h-1)
    peak = peak_idx.unsqueeze(1)  # (B, 1)
    asc = (idx < peak).float()
    desc = (idx >= peak).float()

    eps = 1e-6  # allow tiny numerical noise
    # Ascending region: penalise negative slope (fₚ should be increasing)
    loss_asc = (torch.relu(-diff + eps) * asc).mean()
    # Descending region: penalise positive slope (fₚ should be decreasing)
    loss_desc = (torch.relu(diff + eps) * desc).mean()

    return loss_asc + loss_desc


def background_loss(ne_pred: torch.Tensor, ne_iri: torch.Tensor) -> torch.Tensor:
    """4D-Var analogue background cost: MSE vs IRI prior.

    L_bg = (1/N_h) ‖ Ne_pred − Ne_IRI ‖²

    Parameters
    ----------
    ne_pred : (B, N_h) network prediction
    ne_iri  : (B, N_h) IRI-generated Ne profile (the "background")

    Returns
    -------
    Scalar loss.
    """
    return nn.functional.mse_loss(ne_pred, ne_iri)


# ---------------------------------------------------------------------------
# Combined loss module
# ---------------------------------------------------------------------------


class PhysicsLoss(nn.Module):
    """Combined physics-informed loss for NN-POLAN.

    Stage 1 (supervised)    : abel + background + monotone
    Stage 2 (assimilation)  : abel + monotone  (no IRI background)

    Parameters
    ----------
    lambda_phy  : weight on Abel physics loss
    lambda_mono : weight on monotonicity regulariser
    lambda_bg   : weight on background (IRI) loss; set 0 for Stage 2
    """

    def __init__(
        self,
        lambda_phy: float = 1.0,
        lambda_mono: float = 0.01,
        lambda_bg: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_phy = lambda_phy
        self.lambda_mono = lambda_mono
        self.lambda_bg = lambda_bg

    def forward(
        self,
        ne_pred: torch.Tensor,
        h_virt_obs: torch.Tensor,
        ne_iri: torch.Tensor | None = None,
        obs_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Parameters
        ----------
        ne_pred    : (B, N_h) predicted electron density [cm⁻³]
        h_virt_obs : (B, N_f) observed virtual heights [km]
        ne_iri     : (B, N_h) IRI background (None → lambda_bg has no effect)
        obs_mask   : (B, N_f) bool mask for valid observations

        Returns
        -------
        dict with keys: 'total', 'abel', 'monotone', 'background'
        """
        l_abel = abel_loss(ne_pred, h_virt_obs, obs_mask)
        l_mono = monotone_loss(ne_pred)

        l_bg = torch.zeros(1, device=ne_pred.device, dtype=ne_pred.dtype)
        if ne_iri is not None and self.lambda_bg > 0:
            l_bg = background_loss(ne_pred, ne_iri)

        total = (
            self.lambda_phy * l_abel + self.lambda_mono * l_mono + self.lambda_bg * l_bg
        )

        return {
            "total": total,
            "abel": l_abel,
            "monotone": l_mono,
            "background": l_bg,
        }
