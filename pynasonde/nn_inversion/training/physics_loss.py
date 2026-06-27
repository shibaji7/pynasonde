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
    from pynasonde.nn_inversion.training.physics_loss import (
        PhysicsLoss, torch_forward_batch
    )
    criterion = PhysicsLoss(lambda_phy=1.0, lambda_mono=0.01, lambda_bg=0.1)
    loss = criterion(ne_pred, h_virt_obs, ne_iri=ne_iri_batch)
"""

from __future__ import annotations

import math as _math

import torch
import torch.nn as nn

from pynasonde.nn_inversion.config import NNCfg
from pynasonde.nn_inversion.forward_model import (
    _FP2_CONST,
    _REFLECT_GUARD,
    F_GRID_MHZ,
    H_GRID_KM,
)
from pynasonde.vipir.ngi.utils import load_toml as _load_toml

# ---------------------------------------------------------------------------
# Constants (pre-compute and register as buffers in the loss module)
# ---------------------------------------------------------------------------
_N_H = len(H_GRID_KM)
_N_F = len(F_GRID_MHZ)

# Height spacing (uniform grid assumed)
_DH_KM = float(H_GRID_KM[1] - H_GRID_KM[0])  # 0.5 km
_H_BASE = float(H_GRID_KM[0])  # 60.0 km

# Normalisation scale for the Abel loss — from config.toml
# Dividing h' differences by this brings the MSE from O(h_km²) ≈ O(66000)
# down to O(1), making it numerically comparable to the background loss.
_HV_NORM_KM: float = float(NNCfg.normalisation.hv_norm_km)

# Maximum group refractive index μ' allowed in the differentiable Abel integral.
# Near foF2, μ' = 1/√(1−fp²/f²) → ∞ (singularity).  On a discrete grid this
# produces gradients up to ~200× larger than elsewhere, overwhelming the
# background-loss gradient and driving the Ne profile to collapse in the first
# physics epoch.  Clamping at mu_max (default 50) keeps the Abel gradient
# in the same numerical range as the background gradient while still providing
# a valid physics signal for all sub-foF2 frequencies away from the critical
# point (fp/f ≲ 0.9998 → μ' ≲ 50).
_MU_MAX: float = float(_load_toml().nn_inversion.forward_model.mu_max)

# Pre-compute static tensors on CPU; moved to device in forward
_H_TENSOR = torch.tensor(H_GRID_KM, dtype=torch.float64)  # (N_h,)
_F_TENSOR = torch.tensor(F_GRID_MHZ, dtype=torch.float64)  # (N_f,)

_PI_HALF: float = _math.pi / 2.0


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
    reflect_guard : fp/f amplitude ratio at or above which the layer is
                    treated as reflecting (stops the integral)

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
    fp2 = ne_cm3 / _FP2_CONST  # (B, N_h)

    # Broadcast for all frequencies.
    #   fp2_b : (B, N_h, 1)
    #   f2    : (1,  1,  N_f)
    fp2_b = fp2.unsqueeze(2)  # (B, N_h, 1)
    f2 = (f_grid**2).reshape(1, 1, n_f)  # (1, 1, N_f)
    power_ratio = fp2_b / (f2 + 1e-30)  # fp²/f²  (B, N_h, N_f)

    # group refractive index μ' = 1/√(1 − fₚ²/f²), using power_ratio.
    # Clamp at guard²: reflect_guard is the fp/f amplitude threshold (matching
    # forward_model.py), so the equivalent power-ratio clamp is guard².
    guard2 = reflect_guard**2
    ratio_safe = torch.clamp(power_ratio, max=guard2)
    mu = 1.0 / torch.sqrt(1.0 - ratio_safe + 1e-30)  # (B, N_h, N_f)

    # Cap μ' to prevent singularity-driven gradient explosion near foF2.
    # Without this clamp, heights where fp/f → reflect_guard produce
    # μ' up to ~(1/√(1−guard²)) ≈ 224, yielding Abel gradients ~200×
    # larger than elsewhere and causing Ne collapse in the first physics epoch.
    mu = torch.clamp(mu, max=_MU_MAX)

    # Reflection mask using amplitude ratio fp/f — consistent with forward_batch
    # which tests fp/f >= reflect_guard (not fp²/f²).
    amp_ratio = torch.sqrt(power_ratio.clamp(min=0.0))  # fp/f (B, N_h, N_f)
    above_guard = amp_ratio >= reflect_guard  # (B, N_h, N_f) bool

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
# Abel inversion: h'(f) → Ne(h)
# ---------------------------------------------------------------------------


def abel_invert_batch(
    hv_km: torch.Tensor,  # (B, N_f)  observed h'(f) in km
    obs_mask: torch.Tensor,  # (B, N_f)  bool, True where f < foF2
    n_quad: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert an observed ionogram trace h'(f) to an Ne(h) target profile.

    Uses the singularity-free quadrature identity:

        h_r(f_p) = (2/π) ∫_0^{π/2} h'(f_p sin θ) dθ
                 ≈ (1/n_quad) Σ_k h'(f_p sin θ_k)

    where θ_k are midpoint-rule quadrature nodes.  This substitution removes
    the 1/√(f²−ξ²) singularity that appears in the direct Abel integral kernel,
    making the quadrature uniformly accurate up to f_p = foF2.

    At the true reflection height h_r(f_p) the plasma frequency equals f_p, so:

        Ne(h_r(f_p)) = f_p² × FP2_CONST  [cm⁻³]

    Interpolating these (h_r, Ne) pairs to H_GRID_KM gives the Ne target.

    Parameters
    ----------
    hv_km    : (B, N_f)  observed virtual heights [km]
    obs_mask : (B, N_f)  True where h'(f) is valid (f ≤ foF2)
    n_quad   : number of midpoint-rule quadrature points (default 64)

    Returns
    -------
    ne_target : (B, N_h) float32  Ne [cm⁻³]; 1.0 where undetermined
    btm_mask  : (B, N_h) bool     True for heights within the observed bottomside
    """
    B, N_f = hv_km.shape
    device = hv_km.device
    N_h = _H_TENSOR.shape[0]

    # Work in float64 for numerical precision in the inversion
    hv = hv_km.double()
    f_g = _F_TENSOR.to(device=device)  # (N_f,) float64
    h_g = _H_TENSOR.to(device=device)  # (N_h,) float64

    # --- quadrature nodes on [0, π/2] (midpoint rule) ---
    theta = (torch.arange(n_quad, device=device, dtype=torch.float64) + 0.5) * (
        _PI_HALF / n_quad
    )
    sin_th = torch.sin(theta)  # (Q,)

    # --- interpolation indices for all (f_i, θ_k) pairs ---
    # f_sample[i, k] = f_g[i] * sin(θ_k)
    f_sample = f_g.unsqueeze(1) * sin_th.unsqueeze(0)  # (N_f, Q)
    f_flat = f_sample.reshape(-1).clamp(f_g[0], f_g[-1])  # (N_f*Q,)

    j = torch.searchsorted(f_g.contiguous(), f_flat.contiguous())
    j = j.clamp(1, N_f - 1) - 1  # lower bracket index
    j = j.clamp(0, N_f - 2)  # ensure j+1 is valid

    w = ((f_flat - f_g[j]) / (f_g[j + 1] - f_g[j] + 1e-30)).clamp(0.0, 1.0)  # (N_f*Q,)

    # --- batch-vectorised linear interpolation of hv ---
    hv_lo = hv[:, j]  # (B, N_f*Q)
    hv_hi = hv[:, j + 1]  # (B, N_f*Q)
    hv_sampled = (hv_lo + w * (hv_hi - hv_lo)).reshape(B, N_f, n_quad)  # (B, N_f, Q)

    # h_r[b, i] ≈ (1/n_quad) Σ_k h'(f_i sin θ_k)
    h_r = hv_sampled.mean(dim=-1)  # (B, N_f)

    # Ne at reflection heights: Ne = f_p² × FP2_CONST
    ne_r = f_g**2 * _FP2_CONST  # (N_f,)

    # --- interpolate per-sample (h_r[b, valid], ne_r[valid]) → H_GRID_KM ---
    ne_target = torch.ones(B, N_h, device=device, dtype=torch.float64)
    btm_mask = torch.zeros(B, N_h, device=device, dtype=torch.bool)

    for b in range(B):
        valid = obs_mask[b]
        if not valid.any():
            continue

        hr_b = h_r[b, valid]  # (M,)
        ne_b = ne_r[valid]  # (M,)

        # Sort by height (monotone for well-behaved bottomside)
        sidx = torch.argsort(hr_b)
        hr_s = hr_b[sidx]
        ne_s = ne_b[sidx]

        in_range = (h_g >= hr_s[0]) & (h_g <= hr_s[-1])  # (N_h,)
        if not in_range.any():
            continue

        h_q = h_g[in_range]
        idx = torch.searchsorted(hr_s.contiguous(), h_q.contiguous()).clamp(
            1, hr_s.shape[0] - 1
        )
        h0 = hr_s[idx - 1]
        h1 = hr_s[idx]
        ne0 = ne_s[idx - 1]
        ne1 = ne_s[idx]
        wh = ((h_q - h0) / (h1 - h0 + 1e-30)).clamp(0.0, 1.0)

        ne_target[b, in_range] = ne0 + wh * (ne1 - ne0)
        btm_mask[b, in_range] = True

    return ne_target.to(hv_km.dtype), btm_mask


# ---------------------------------------------------------------------------
# Individual loss functions
# ---------------------------------------------------------------------------


def abel_loss(
    ne_pred: torch.Tensor,
    h_virt_obs: torch.Tensor,
    obs_mask: torch.Tensor | None = None,
    mode: str = "mse",
) -> torch.Tensor:
    """Physics consistency loss: Abel(Ne_pred) vs observed h'(f).

    Three formulations are available via ``mode``:

    ``"mse"``  (default)
        Standard MSE on normalised absolute differences::

            L = mean( ((h'_pred − h'_obs) / HV_NORM)² )

        Simple and interpretable but the gradient magnitude diverges near
        foF2 (h'→∞) and for no-reflection frequencies, causing the
        gradient-explosion observed in Config B of diagnose_convergence.py.

    ``"log"``
        MSE on the log ratio of virtual heights::

            L = mean( log²( h'_pred / h'_obs ) )

        The log ratio is bounded even when h'_pred → grid_top (no
        reflection): log(512/200) ≈ 0.94 vs MSE term (312/150)² ≈ 4.3.
        Gradient ∂L/∂h'_pred = 2·log(r)/h'_pred stays finite as
        h'_pred → ∞, structurally capping the singularity-driven
        gradient explosion without needing a trim mask.

    ``"cumulative"``
        MSE on the running frequency-integral of virtual height::

            L = mean( (∫₀ᶠ h'_pred df' − ∫₀ᶠ h'_obs df')² ) / scale²

        A spike at one near-foF2 frequency adds a constant offset to all
        higher-frequency cumulative sums, so the gradient is spread across
        all frequencies rather than concentrated at one singularity point.
        Naturally damps the near-foF2 gradient spikes.

    Parameters
    ----------
    ne_pred    : (B, N_h) predicted electron density [cm⁻³]
    h_virt_obs : (B, N_f) observed (or IRI-generated) virtual heights [km]
    obs_mask   : (B, N_f) bool — True where h_virt_obs is valid
    mode       : one of ``"mse"``, ``"log"``, ``"cumulative"``

    Returns
    -------
    Scalar loss value.
    """
    if mode not in ("mse", "log", "cumulative"):
        raise ValueError(
            f"abel_loss mode must be 'mse', 'log', or 'cumulative'; got {mode!r}"
        )

    h_virt_pred = torch_forward_batch(ne_pred.double()).to(ne_pred.dtype)  # (B, N_f)

    mask_f = obs_mask.float() if obs_mask is not None else torch.ones_like(h_virt_pred)
    n_valid = mask_f.sum().clamp(min=1.0)

    if mode == "mse":
        diff = (h_virt_pred - h_virt_obs) / _HV_NORM_KM
        diff = diff * mask_f
        return (diff**2).sum() / n_valid

    elif mode == "log":
        # log( h'_pred / h'_obs ) — bounded even when h'_pred → grid_top
        ratio = h_virt_pred / h_virt_obs.clamp(min=1.0)
        log_diff = torch.log(ratio.clamp(min=1e-6))  # (B, N_f)
        log_diff = log_diff * mask_f
        return (log_diff**2).sum() / n_valid

    else:  # cumulative
        # Running trapezoidal integral along the frequency axis.
        # Scale by HV_NORM_KM * N_f * df so the loss is O(1).
        df = _F_TENSOR[1] - _F_TENSOR[0]  # 0.1 MHz (scalar tensor)
        cum_pred = torch.cumsum(h_virt_pred * mask_f, dim=1) * df  # (B, N_f)
        cum_obs = torch.cumsum(h_virt_obs * mask_f, dim=1) * df  # (B, N_f)
        n_f = h_virt_pred.shape[1]
        scale = _HV_NORM_KM * n_f * df.item()
        diff = (cum_pred - cum_obs) / scale
        diff = diff * mask_f
        return (diff**2).sum() / n_valid


def monotone_loss(ne_pred: torch.Tensor) -> torch.Tensor:
    """Penalise non-monotone decay in the topside plasma-frequency profile.

    The topside (above hmF2) is constrained by diffusive equilibrium to
    decrease monotonically.  The Abel integral provides no information above
    foF2 (obs_mask = 0 there), so without an explicit regulariser the network
    can freely oscillate in that region.

    The bottomside (below hmF2) is intentionally NOT penalised here:
      • the Abel loss already constrains the E/F1/F2 layer structure, and
      • real IRI profiles are multi-layer (E + F1 + F2) and thus NOT
        monotone below the F2 peak — penalising them would create a
        systematic conflict with the background loss.

    L_mono = mean( ReLU(+Δfₚ + ε)  for h > h_peak )

    where ε is a small tolerance to allow grid-scale numerical noise.

    Parameters
    ----------
    ne_pred : (B, N_h) — we compute fₚ² ∝ Ne

    Returns
    -------
    Scalar loss (topside monotone decreasing only).
    """
    fp2 = ne_pred / _FP2_CONST  # (B, N_h)  fₚ²[MHz²]
    diff = fp2[:, 1:] - fp2[:, :-1]  # (B, N_h-1) — positive = ascending

    # Find the F2 peak index per profile
    peak_idx = fp2.argmax(dim=1)  # (B,)

    # Descending (topside) mask: indices at or above the peak
    n_h = fp2.shape[1]
    idx = torch.arange(n_h - 1, device=ne_pred.device).unsqueeze(0)  # (1, N_h-1)
    peak = peak_idx.unsqueeze(1)  # (B, 1)
    desc = (idx >= peak).float()  # 1 for topside grid cells

    eps = 1e-6  # allow tiny numerical noise
    # Topside only: penalise any positive slope (fₚ should be decreasing)
    loss_desc = (torch.relu(diff + eps) * desc).mean()

    return loss_desc


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
        abel_mode: str = "mse",
    ) -> None:
        super().__init__()
        self.lambda_phy = lambda_phy
        self.lambda_mono = lambda_mono
        self.lambda_bg = lambda_bg
        self.abel_mode = abel_mode  # "mse" | "log" | "cumulative"

    def forward(
        self,
        ne_pred: torch.Tensor,
        h_virt_obs: torch.Tensor | None = None,
        ne_iri: torch.Tensor | None = None,
        obs_mask: torch.Tensor | None = None,
        ne_pred_n: torch.Tensor | None = None,
        ne_abel_target: torch.Tensor | None = None,
        btm_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Parameters
        ----------
        ne_pred        : (B, N_h) predicted electron density [cm⁻³]
        h_virt_obs     : (B, N_f) observed virtual heights [km].
                         Used for the forward Abel loss when ne_abel_target
                         is not provided.
        ne_iri         : (B, N_h) IRI background in *normalised log* space.
                         Pass None → lambda_bg has no effect.
        obs_mask       : (B, N_f) bool mask — used with forward Abel loss only.
        ne_pred_n      : (B, N_h) network output in normalised log Ne space.
                         When provided, bg loss = MSE(ne_pred_n, ne_iri).
        ne_abel_target : (B, N_h) Ne [cm⁻³] from Abel inversion of h'_obs.
                         When provided, Abel loss = log10-MSE(ne_pred, ne_abel)
                         on the bottomside only (no forward model in the graph).
        btm_mask       : (B, N_h) bool — valid bottomside heights from
                         abel_invert_batch; used to restrict the Abel loss.

        Returns
        -------
        dict with keys: 'total', 'abel', 'monotone', 'background'
        """
        _zero = torch.zeros(1, device=ne_pred.device, dtype=ne_pred.dtype)

        if self.lambda_phy > 0:
            if ne_abel_target is not None:
                # Option 3: Ne-space inversion loss — no forward Abel model,
                # gradient is well-conditioned everywhere.
                log_pred = torch.log10(ne_pred.clamp(min=1.0))
                log_abel = torch.log10(ne_abel_target.to(ne_pred.dtype).clamp(min=1.0))
                if btm_mask is not None:
                    mf = btm_mask.float()
                    n_valid = mf.sum().clamp(min=1.0)
                    l_abel = ((log_pred - log_abel) ** 2 * mf).sum() / n_valid
                else:
                    l_abel = nn.functional.mse_loss(log_pred, log_abel)
            else:
                # Original forward Abel loss (kept for diagnostics / ablations)
                l_abel = abel_loss(ne_pred, h_virt_obs, obs_mask, mode=self.abel_mode)
        else:
            l_abel = _zero

        l_mono = monotone_loss(ne_pred) if self.lambda_mono > 0 else _zero
        l_bg = _zero
        if ne_iri is not None and self.lambda_bg > 0:
            pred_for_bg = ne_pred_n if ne_pred_n is not None else ne_pred
            l_bg = background_loss(pred_for_bg, ne_iri)

        total = (
            self.lambda_phy * l_abel + self.lambda_mono * l_mono + self.lambda_bg * l_bg
        )

        return {
            "total": total,
            "abel": l_abel,
            "monotone": l_mono,
            "background": l_bg,
        }
