"""Zero-dependency numpy inference engine for NN-POLAN.

Loads weights exported by export_weights.py and runs forward inference
as pure numpy matrix operations — no PyTorch or iricore required at runtime.

This module mirrors the architecture in training/architecture.py exactly.
Any changes to the architecture must be reflected here.

Supported activations: SiLU, softplus (used in the trained model).
BatchNorm is applied in eval mode (uses running mean/var).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


# ---------------------------------------------------------------------------
# Layer primitives
# ---------------------------------------------------------------------------


def _bn_eval(x: np.ndarray, w: dict, prefix: str) -> np.ndarray:
    """BatchNorm1d in eval mode.

    x : (B, C, L) or (B, C)
    """
    gamma = (
        w[f"{prefix}.weight"][:, np.newaxis] if x.ndim == 3 else w[f"{prefix}.weight"]
    )
    beta = w[f"{prefix}.bias"][:, np.newaxis] if x.ndim == 3 else w[f"{prefix}.bias"]
    mean = (
        w[f"{prefix}.running_mean"][:, np.newaxis]
        if x.ndim == 3
        else w[f"{prefix}.running_mean"]
    )
    var = (
        w[f"{prefix}.running_var"][:, np.newaxis]
        if x.ndim == 3
        else w[f"{prefix}.running_var"]
    )
    return gamma * (x - mean) / np.sqrt(var + 1e-5) + beta


def _conv1d(
    x: np.ndarray, w: dict, prefix: str, stride: int = 1, padding: int = 0
) -> np.ndarray:
    """Conv1d forward pass. x: (B, C_in, L)."""
    kernel = w[f"{prefix}.weight"]  # (C_out, C_in, K)
    bias = w.get(f"{prefix}.bias")  # (C_out,) or None
    C_out, C_in, K = kernel.shape
    B, _, L = x.shape

    if padding > 0:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="constant")

    L_out = (x.shape[2] - K) // stride + 1
    out = np.zeros((B, C_out, L_out), dtype=x.dtype)

    # Naive O(B·C_out·C_in·K·L_out) — acceptable for inference (small batches)
    for k in range(K):
        out += (
            kernel[:, :, k][np.newaxis].transpose(0, 2, 1)
            @ x[:, :, k::stride][:, :, :L_out]
        )
    if bias is not None:
        out += bias[np.newaxis, :, np.newaxis]
    return out


def _conv_transpose1d(
    x: np.ndarray,
    w: dict,
    prefix: str,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
) -> np.ndarray:
    """ConvTranspose1d — implemented as fractional-stride conv."""
    kernel = w[f"{prefix}.weight"]  # (C_in, C_out, K)  ← transposed storage
    bias = w.get(f"{prefix}.bias")
    C_in_w, C_out, K = kernel.shape
    B, C_in, L = x.shape

    L_out = (L - 1) * stride - 2 * padding + K + output_padding
    out = np.zeros((B, C_out, L_out), dtype=x.dtype)

    for b in range(B):
        for c_in in range(C_in):
            for c_out in range(C_out):
                for i in range(L):
                    start = i * stride
                    out[b, c_out, start : start + K] += (
                        x[b, c_in, i] * kernel[c_in, c_out, :]
                    )

    if bias is not None:
        out += bias[np.newaxis, :, np.newaxis]
    return out


def _linear(x: np.ndarray, w: dict, prefix: str) -> np.ndarray:
    """Linear layer. x: (B, in) → (B, out)."""
    weight = w[f"{prefix}.weight"]  # (out, in)
    bias = w.get(f"{prefix}.bias")  # (out,)
    out = x @ weight.T
    if bias is not None:
        out += bias[np.newaxis]
    return out


def _film_conv_block(
    x: np.ndarray,
    w: dict,
    prefix: str,
    gamma: np.ndarray,
    beta: np.ndarray,
    transpose: bool = False,
    stride: int = 1,
    padding: int = 1,
    output_padding: int = 0,
) -> np.ndarray:
    """FiLM-conditioned conv block (conv → bn → FiLM → SiLU).

    gamma, beta : (B, C, 1)
    """
    if transpose:
        x = _conv_transpose1d(
            x,
            w,
            f"{prefix}.conv",
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
    else:
        x = _conv1d(x, w, f"{prefix}.conv", stride=stride, padding=padding)
    x = _bn_eval(x, w, f"{prefix}.bn")
    x = gamma * x + beta
    return _silu(x)


# ---------------------------------------------------------------------------
# FiLM generator
# ---------------------------------------------------------------------------


def _film_generator(
    c: np.ndarray, w: dict, prefix: str, n_layers: int, feat_dim: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward pass of FiLMGenerator MLP.

    c : (B, 6)
    Returns list of (gamma, beta) each (B, feat_dim, 1)
    """
    x = _silu(_linear(c, w, f"{prefix}.mlp.0"))
    x = _silu(_linear(x, w, f"{prefix}.mlp.2"))
    x = _linear(x, w, f"{prefix}.mlp.4")  # (B, 2*n_layers*feat_dim)
    B = c.shape[0]
    x = x.reshape(B, n_layers, 2, feat_dim)
    result = []
    for i in range(n_layers):
        gamma = (1.0 + x[:, i, 0, :])[:, :, np.newaxis]  # (B, D, 1)
        beta = x[:, i, 1, :][:, :, np.newaxis]  # (B, D, 1)
        result.append((gamma, beta))
    return result


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

_ENC_CONFIGS = [
    # (in_ch, out_ch, kernel, stride, padding, transpose, out_pad)
    (1, 32, 7, 1, 3, False, 0),  # block 0
    (32, 64, 3, 2, 1, False, 0),  # block 1 (stride-2)
    (64, 64, 3, 1, 1, False, 0),  # block 2
    (64, 128, 3, 2, 1, False, 0),  # block 3 (stride-2)
    (128, 128, 3, 1, 1, False, 0),  # block 4
    (128, 128, 3, 2, 1, False, 0),  # block 5 (stride-2)
    (128, 128, 3, 1, 1, False, 0),  # block 6
    (128, 128, 3, 2, 1, False, 0),  # block 7 (stride-2)
]


def _encoder(x: np.ndarray, w: dict, film_params: list, feat_dim: int) -> np.ndarray:
    """x: (B, 1, N_f) → (B, latent_dim)."""
    for i, (_, _, _, stride, padding, transpose, _) in enumerate(_ENC_CONFIGS):
        gamma, beta = film_params[i]
        x = _film_conv_block(
            x,
            w,
            f"encoder.blocks.{i}",
            gamma,
            beta,
            transpose=False,
            stride=stride,
            padding=padding,
        )
    # AdaptiveAvgPool1d(1) → squeeze → linear
    x = x.mean(axis=2)  # (B, feat_dim)
    x = _linear(x, w, "encoder.fc_out")
    return x


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

_DEC_CONFIGS = [
    # (in_ch, out_ch, kernel, stride, padding, transpose, out_pad)
    (128, 128, 3, 2, 1, True, 1),  # block 0  9→18
    (128, 128, 3, 1, 1, False, 0),  # block 1
    (128, 128, 3, 2, 1, True, 1),  # block 2  18→36
    (128, 64, 3, 1, 1, False, 0),  # block 3
    (64, 64, 3, 2, 1, True, 1),  # block 4  36→72
    (64, 32, 3, 1, 1, False, 0),  # block 5
    (32, 32, 3, 2, 1, True, 1),  # block 6  72→144
    (32, 32, 3, 1, 1, False, 0),  # block 7
]

_DEC_INIT_LEN = 9


def _decoder(
    z: np.ndarray, w: dict, film_params: list, feat_dim: int, n_h: int
) -> np.ndarray:
    """z: (B, latent_dim) → (B, N_h)."""
    B = z.shape[0]
    x = _linear(z, w, "decoder.fc_in")  # (B, feat_dim * init_len)
    x = x.reshape(B, feat_dim, _DEC_INIT_LEN)  # (B, feat_dim, 9)

    for i, (_, _, _, stride, padding, transpose, out_pad) in enumerate(_DEC_CONFIGS):
        gamma, beta = film_params[i]
        x = _film_conv_block(
            x,
            w,
            f"decoder.blocks.{i}",
            gamma,
            beta,
            transpose=transpose,
            stride=stride,
            padding=padding,
            output_padding=out_pad,
        )

    # proj 1×1 + adaptive pool to n_h
    x = _conv1d(x, w, "decoder.proj", stride=1, padding=0)  # (B, 1, L)
    x = x.squeeze(1)  # (B, L)

    # AdaptiveAvgPool1d(n_h) — average-pool to target length
    L = x.shape[1]
    if L != n_h:
        # Fractional pooling via reshape + mean
        x = _adaptive_avg_pool1d(x, n_h)

    return _softplus(x)


def _adaptive_avg_pool1d(x: np.ndarray, n_out: int) -> np.ndarray:
    """Mimics PyTorch AdaptiveAvgPool1d for 2-D input (B, L)."""
    B, L = x.shape
    out = np.zeros((B, n_out), dtype=x.dtype)
    for i in range(n_out):
        start = int(np.floor(i * L / n_out))
        end = int(np.ceil((i + 1) * L / n_out))
        out[:, i] = x[:, start:end].mean(axis=1)
    return out


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class NNPolanNumpy:
    """Pure-numpy NN-POLAN inference.

    Load once, call many times — thread-safe (read-only weights).

    Parameters
    ----------
    npz_path : path to the .npz weight file produced by export_weights.py
    """

    def __init__(self, npz_path: str | Path) -> None:
        data = np.load(npz_path, allow_pickle=False)
        self._w = dict(data)

        self.h_grid_km = self._w["__h_grid_km"]
        self.f_grid_mhz = self._w["__f_grid_mhz"]
        self.cond_mean = self._w["__cond_mean"]
        self.cond_std = self._w["__cond_std"]
        self.hv_mean = float(self._w["__hv_mean"][0])
        self.hv_std = float(self._w["__hv_std"][0])
        self.log_ne_min = float(self._w["__log_ne_min"][0])
        self.log_ne_max = float(self._w["__log_ne_max"][0])
        self.latent_dim = int(self._w["__latent_dim"][0])
        self.feat_dim = int(self._w["__feat_dim"][0])
        self._n_h = len(self.h_grid_km)

    def _norm_cond(self, c: np.ndarray) -> np.ndarray:
        return (c - self.cond_mean) / (self.cond_std + 1e-8)

    def _norm_hv(self, hv: np.ndarray) -> np.ndarray:
        return (hv - self.hv_mean) / self.hv_std

    def _denorm_ne(self, ne_n: np.ndarray) -> np.ndarray:
        log_ne = ne_n * (self.log_ne_max - self.log_ne_min) + self.log_ne_min
        return 10.0**log_ne

    def predict(
        self,
        h_virtual_km: np.ndarray,
        cond: np.ndarray,
    ) -> np.ndarray:
        """Invert virtual-height trace to electron density profile.

        Parameters
        ----------
        h_virtual_km : (B, N_f) or (N_f,) — virtual heights [km]
        cond         : (B, 6)  or (6,)    — [lat, lon, doy, ut, Kp, F10.7]

        Returns
        -------
        ne_cm3 : (B, N_h) or (N_h,) — electron density [cm⁻³]
        """
        squeeze = h_virtual_km.ndim == 1
        if squeeze:
            h_virtual_km = h_virtual_km[np.newaxis]
            cond = cond[np.newaxis]

        h_virtual_km = h_virtual_km.astype(np.float32)
        cond = cond.astype(np.float32)

        hv_n = self._norm_hv(h_virtual_km)  # (B, N_f)
        cond_n = self._norm_cond(cond)  # (B, 6)

        w = self._w
        fd = self.feat_dim

        # FiLM params
        film_enc = _film_generator(cond_n, w, "film_enc", 8, fd)
        film_dec = _film_generator(cond_n, w, "film_dec", 8, fd)

        # Encoder
        x = hv_n[:, np.newaxis, :]  # (B, 1, N_f)
        z = _encoder(x, w, film_enc, fd)

        # Decoder
        ne_n = _decoder(z, w, film_dec, fd, self._n_h)  # (B, N_h)

        ne = self._denorm_ne(ne_n)

        return ne.squeeze(0) if squeeze else ne
