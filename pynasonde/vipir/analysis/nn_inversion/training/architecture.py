"""NN-POLAN model architecture.

Two-stage physics-informed ionospheric inversion network.

Architecture overview
---------------------

Input  : h'(f)  — virtual-height trace  (N_f,)
Output : N(h)   — electron density      (N_h,) in cm⁻³

Encoder
    1-D CNN operating on the virtual-height trace.
    Each conv block is FiLM-conditioned by the geophysical context vector c.

Decoder
    Symmetric 1-D transposed-conv stack expanding latent code → N_h.
    Constrained output: softplus activation ensures N(h) ≥ 0.
    Optional Chapman-shape regularisation via skip connections.

FiLM conditioning (Feature-wise Linear Modulation)
    Context  c = [lat, lon, doy, ut, Kp, F10.7]  →  MLP  →  (γ, β) per layer.
    Applied after every BatchNorm in both encoder and decoder.

Shapes (using default grid sizes from forward_model.py)
    N_f = 141  (F_GRID_MHZ)
    N_h = 226  (H_GRID_KM)
    C_dim = 6  (COND_COLS)

Usage
-----
    from pynasonde.vipir.analysis.nn_inversion.training.architecture import NNPolan
    model = NNPolan()
    ne_pred = model(h_virtual, cond)   # (B, N_h)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pynasonde.vipir.analysis.nn_inversion.forward_model import F_GRID_MHZ, H_GRID_KM

_N_F: int = len(F_GRID_MHZ)  # 141
_N_H: int = len(H_GRID_KM)  # 904  (60–511.5 km at 0.5 km step)
_C_DIM: int = 6  # conditioning-vector dimension


# ---------------------------------------------------------------------------
# FiLM conditioning network
# ---------------------------------------------------------------------------


class FiLMGenerator(nn.Module):
    """Maps conditioning vector c → (γ, β) pairs for each FiLM-modulated layer.

    Each layer gets its own output head so (γ, β) are always the same size as
    that layer's feature map — avoids broadcast mismatches when channel counts vary.

    Parameters
    ----------
    c_dim        : dimension of conditioning vector (default 6)
    hidden_dim   : MLP shared trunk width
    channel_dims : list of out_channels for each conditioned layer
                   (one entry per CNN block, in forward order)
    """

    def __init__(
        self,
        c_dim: int = _C_DIM,
        hidden_dim: int = 128,
        channel_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        if channel_dims is None:
            channel_dims = [128] * 8

        # Shared MLP trunk
        self.trunk = nn.Sequential(
            nn.Linear(c_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        # Per-layer projection heads: each outputs 2*ch values (γ and β)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 2 * ch) for ch in channel_dims]
        )

    def forward(self, c: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return list of (γ, β) tensors, one per conditioned layer.

        Parameters
        ----------
        c : (B, C_dim)

        Returns
        -------
        list of length len(channel_dims), each element (γ, β) of shape (B, ch, 1)
        where ch matches the corresponding CNN block's out_channels.
        """
        h = self.trunk(c)  # (B, hidden_dim)
        result = []
        for head in self.heads:
            out = head(h)  # (B, 2*ch)
            ch = out.shape[1] // 2
            gamma = 1.0 + out[:, :ch].unsqueeze(-1)  # (B, ch, 1) centred at 1
            beta = out[:, ch:].unsqueeze(-1)  # (B, ch, 1)
            result.append((gamma, beta))
        return result


# ---------------------------------------------------------------------------
# FiLM-conditioned conv block
# ---------------------------------------------------------------------------


class FiLMConvBlock(nn.Module):
    """Conv1d → BatchNorm → FiLM → SiLU."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
        transpose: bool = False,
        output_padding: int = 0,
    ) -> None:
        super().__init__()
        if transpose:
            self.conv = nn.ConvTranspose1d(
                in_ch,
                out_ch,
                kernel,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            )
        else:
            self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        film: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = self.bn(self.conv(x))
        if film is not None:
            gamma, beta = film
            x = gamma * x + beta
        return self.act(x)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """1-D CNN encoder: (B, 1, N_f) → (B, latent_dim).

    Progressively halves the sequence length with stride-2 convolutions,
    then global-average-pools to a fixed-size latent vector.
    """

    def __init__(self, latent_dim: int = 256, feat_dim: int = 128) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.blocks = nn.ModuleList(
            [
                FiLMConvBlock(1, feat_dim // 4, kernel=7, padding=3),  # layer 0
                FiLMConvBlock(
                    feat_dim // 4, feat_dim // 2, stride=2, padding=1
                ),  # layer 1 (÷2)
                FiLMConvBlock(feat_dim // 2, feat_dim // 2),  # layer 2
                FiLMConvBlock(
                    feat_dim // 2, feat_dim, stride=2, padding=1
                ),  # layer 3 (÷2)
                FiLMConvBlock(feat_dim, feat_dim),  # layer 4
                FiLMConvBlock(feat_dim, feat_dim, stride=2, padding=1),  # layer 5 (÷2)
                FiLMConvBlock(feat_dim, feat_dim),  # layer 6
                FiLMConvBlock(feat_dim, feat_dim, stride=2, padding=1),  # layer 7 (÷2)
            ]
        )
        # After 4 stride-2 layers: N_f=141 → 71 → 36 → 18 → 9
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(feat_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        film_params: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (B, 1, N_f) — virtual-height trace
        film_params: list of (γ, β) from FiLMGenerator, length = len(self.blocks)

        Returns
        -------
        z : (B, latent_dim)
        """
        for i, block in enumerate(self.blocks):
            film = film_params[i] if film_params is not None else None
            x = block(x, film)
        x = self.pool(x).squeeze(-1)  # (B, feat_dim)
        return self.fc_out(x)  # (B, latent_dim)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class Decoder(nn.Module):
    """Latent vector → N(h) profile.

    Expands (B, latent_dim) → (B, N_h) electron density.
    Uses transposed convolutions symmetric to the encoder.
    Output is constrained to ≥ 0 via softplus.
    """

    def __init__(self, latent_dim: int = 256, feat_dim: int = 128) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        # Start from a small spatial extent then expand
        self._init_len = (
            9  # must match encoder output spatial size after pooling unrolled
        )

        self.fc_in = nn.Linear(latent_dim, feat_dim * self._init_len)

        self.blocks = nn.ModuleList(
            [
                FiLMConvBlock(
                    feat_dim,
                    feat_dim,
                    transpose=True,
                    stride=2,  # layer 0: 9→18
                    padding=1,
                    output_padding=1,
                ),
                FiLMConvBlock(feat_dim, feat_dim, transpose=False),  # layer 1
                FiLMConvBlock(
                    feat_dim,
                    feat_dim,
                    transpose=True,
                    stride=2,  # layer 2: 18→36
                    padding=1,
                    output_padding=1,
                ),
                FiLMConvBlock(feat_dim, feat_dim // 2, transpose=False),  # layer 3
                FiLMConvBlock(
                    feat_dim // 2,
                    feat_dim // 2,
                    transpose=True,
                    stride=2,  # layer 4: 36→72
                    padding=1,
                    output_padding=1,
                ),
                FiLMConvBlock(feat_dim // 2, feat_dim // 4, transpose=False),  # layer 5
                FiLMConvBlock(
                    feat_dim // 4,
                    feat_dim // 4,
                    transpose=True,
                    stride=2,  # layer 6: 72→144
                    padding=1,
                    output_padding=1,
                ),
                FiLMConvBlock(feat_dim // 4, feat_dim // 4, transpose=False),  # layer 7
            ]
        )

        # Final 1×1 projection; then linear interpolation → exactly N_h points
        self.proj = nn.Conv1d(feat_dim // 4, 1, kernel_size=1)

    def forward(
        self,
        z: torch.Tensor,
        film_params: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z          : (B, latent_dim)
        film_params: list of (γ, β), length = len(self.blocks)

        Returns
        -------
        ne : (B, N_h) in cm⁻³ (all ≥ 0)
        """
        B = z.shape[0]
        x = self.fc_in(z).reshape(B, self.feat_dim, self._init_len)  # (B, C, 9)
        for i, block in enumerate(self.blocks):
            film = film_params[i] if film_params is not None else None
            x = block(x, film)
        x = self.proj(x)  # (B, 1, ~144)
        x = F.interpolate(
            x, size=_N_H, mode="linear", align_corners=True
        )  # (B, 1, N_h)
        x = x.squeeze(1)  # (B, N_h)
        return F.softplus(x)  # Ne ≥ 0


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class NNPolan(nn.Module):
    """Physics-informed NN-POLAN inversion model.

    Parameters
    ----------
    latent_dim : size of the bottleneck latent vector
    feat_dim   : base channel width for encoder/decoder conv blocks
    c_dim      : conditioning-vector dimension (default 6)
    film_hidden: hidden width for the FiLM generator MLP
    """

    def __init__(
        self,
        latent_dim: int = 256,
        feat_dim: int = 128,
        c_dim: int = _C_DIM,
        film_hidden: int = 128,
    ) -> None:
        super().__init__()
        fd = feat_dim
        # out_channels per block, must match Encoder.blocks / Decoder.blocks order
        enc_channels = [fd // 4, fd // 2, fd // 2, fd, fd, fd, fd, fd]
        dec_channels = [fd, fd, fd, fd // 2, fd // 2, fd // 4, fd // 4, fd // 4]

        self.film_enc = FiLMGenerator(c_dim, film_hidden, enc_channels)
        self.film_dec = FiLMGenerator(c_dim, film_hidden, dec_channels)
        self.encoder = Encoder(latent_dim, feat_dim)
        self.decoder = Decoder(latent_dim, feat_dim)

    def forward(
        self,
        h_virtual: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h_virtual : (B, N_f)  virtual-height trace [km]
        cond      : (B, 6)    conditioning vector [lat, lon, doy, ut, Kp, F10.7]

        Returns
        -------
        ne : (B, N_h)  electron density [cm⁻³]
        """
        x = h_virtual.unsqueeze(1)  # (B, 1, N_f) — channel dim for Conv1d
        film_enc = self.film_enc(cond)
        film_dec = self.film_dec(cond)
        z = self.encoder(x, film_enc)  # (B, latent_dim)
        ne = self.decoder(z, film_dec)  # (B, N_h)
        return ne

    # ------------------------------------------------------------------
    # Convenience: parameter count
    # ------------------------------------------------------------------
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
