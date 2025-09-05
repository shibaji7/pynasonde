import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

C = 299_792_458.0  # speed of light [m/s]


def capon_range_spectrum(tx, rx, fs, t_bit, Z=100, K=10, load=1e-3):
    """
    Compute a high-resolution range spectrum for a single frequency point
    using cross-spectrum analysis + Capon (MVDR).

    Parameters
    ----------
    tx : (N,) complex64/128
        Baseband transmit reference sequence for one pulse (same sampling as rx).
        This can be the known code (biphase complementary code, etc.) after the TX chain,
        time-aligned to the captured echo window.
    rx : (N,) complex
        Baseband received echo for one sounding at the same sample rate.
    fs : float
        Sampling rate (Hz). For the paper’s configuration, fs == bandwidth == 1/t_bit.
    t_bit : float
        Code bit duration (seconds). Intrinsic range resolution r0 = c * t_bit / 2.
    Z : int
        Number of subbands (rows of the spectrum matrix). Typical 50–100.
    K : int
        Resolution improvement factor (gives r0/K grid spacing).
    load : float
        Diagonal loading factor for R_f to keep it invertible.

    Returns
    -------
    r_grid_m : (K*V,) float
        Range grid in meters (0 .. (K*V-1)*r0/K).
    P : (K*V,) float
        Capon pseudospectrum (unitless), larger => stronger scatterer at that range.
    """
    # 1) Cross-power spectrum G(f) = S_T(f) * conj(S_Echo(f))  [Eq. 2–3]
    # Use an FFT length that keeps frequency bin spacing = fs / V (V = Nfft)
    N = int(2 ** np.ceil(np.log2(len(tx))))
    ST = np.fft.fft(tx, n=N)
    SE = np.fft.fft(rx, n=N)
    G = ST * np.conj(SE)  # complex cross-spectrum

    # 2) Build spectrum matrix with Z subbands (Hankel-like)  [Eq. 7–8]
    V = N  # number of frequency bins over the bandwidth
    if Z >= V:
        raise ValueError(
            f"Z must be < number of spectrum bins (V={V}). Choose smaller Z."
        )
    # columns = V - Z + 1
    Gm = sliding_window_view(G, Z).T  # shape (Z, V-Z+1)

    # 3) Covariance across subbands  [Eq. 11]
    Rf = (Gm @ Gm.conj().T) / (V - Z + 1)
    # Diagonal loading to avoid singularity when Z is large / SNR low (paper warns about too-large Z)
    # (cf. the deterioration for Z=150)
    Rf = Rf + (load * np.trace(Rf).real / Z) * np.eye(Z)

    # Precompute inverse (use pseudo-inverse for numerical stability)
    Rf_inv = np.linalg.pinv(Rf, rcond=1e-6)

    # 4) Steering vectors for range grid, MVDR/Capon spectrum  [Eq. 10, 12–14]
    r0 = C * t_bit / 2.0  # intrinsic range resolution
    df = fs / V  # frequency bin spacing [Eq. 7]
    dphi = 4.0 * np.pi * df / C  # phase step per subband per meter: Δφ = 4π Δf r / c
    kz = np.arange(Z, dtype=np.float64)  # subband row index (0..Z-1)

    L = K * V  # number of range grid points [Eq. 14]
    r_grid_m = (r0 / K) * np.arange(L)  # 0, r0/K, 2*r0/K, ...
    # Build steering vectors a(r) = [1, e^{j dphi*r}, e^{j 2 dphi*r}, ..., e^{j (Z-1) dphi*r}]^T
    # Vectorized computation for speed:
    phase = np.outer(kz, r_grid_m) * dphi
    A = np.exp(1j * phase)  # shape (Z, L)

    # Capon pseudospectrum P(r) = 1 / (a^H R^-1 a)
    # Compute efficiently with matrix operations:
    ARi = A.conj().T @ Rf_inv  # (L, Z)
    denom = np.einsum("ij,ij->i", ARi, A.conj().T)  # sum over Z
    P = 1.0 / np.maximum(denom.real, 1e-12)
    # Normalize for plotting
    P = (P - P.min()) / max(P.ptp(), 1e-12)
    return r_grid_m, P


def make_ionogram(tx_list, rx_list, fs, t_bit, Z=100, K=10, load=1e-3):
    """
    Process a sweep: list/array of frequency points (tx, rx per freq).
    Returns (freqs, ranges_km, image) ready to plot.
    """
    assert len(tx_list) == len(rx_list)
    nF = len(tx_list)
    # First call to get grid length
    r_grid_m, P0 = capon_range_spectrum(
        tx_list[0], rx_list[0], fs, t_bit, Z=Z, K=K, load=load
    )
    iono = np.zeros((len(r_grid_m), nF), dtype=np.float32)
    iono[:, 0] = P0
    for i in range(1, nF):
        _, P = capon_range_spectrum(
            tx_list[i], rx_list[i], fs, t_bit, Z=Z, K=K, load=load
        )
        iono[:, i] = P
    return r_grid_m / 1000.0, iono  # ranges in km


# ------------------ Example usage ------------------
if __name__ == "__main__":
    """
    Replace the synthetic example below with your real sweep:
      - tx_list[f] : complex array, the known transmit reference for that frequency
      - rx_list[f] : complex array, the recorded echo (one or coherently integrated pulses)
      - fs         : sampling rate AFTER DDC (Hz), typically == 1/t_bit
      - t_bit      : code bit duration (s)  (e.g., 25.6e-6 -> r0 = 3.84 km)
    """
    import matplotlib.pyplot as plt

    # Synthetic demo (single freq): a pair of reflectors ~110 & 112 km
    fs = 39_062.5  # Hz (paper’s example; equals bandwidth)  [WISS]
    t_bit = 25.6e-6  # s  -> r0=3.84 km
    N = 4096
    t = np.arange(N) / fs
    # Transmit: simple biphase-like sequence mapped to +/-1 baseband (toy)
    np.random.seed(0)
    code = 2 * (np.random.randint(0, 2, size=N) - 0.5) + 0j
    tx = code.astype(np.complex64)

    # Echo: two delayed replicas (110 & 112 km) + noise
    def delay_samples(r_km):  # two-way delay
        tau = 2 * (r_km * 1000.0) / C
        return int(round(tau * fs))

    rx = np.zeros_like(tx, dtype=complex)
    for r_km, amp in [(110.0, 1.0), (112.0, 0.7)]:
        d = delay_samples(r_km)
        if d < N:
            rx[d:] += amp * tx[: N - d]
    rx += 0.2 * (np.random.randn(N) + 1j * np.random.randn(N))

    r_km, P = capon_range_spectrum(tx, rx, fs, t_bit, Z=100, K=10, load=5e-3)
    plt.figure(figsize=(7, 3))
    plt.plot(r_km / 1e3, P)
    plt.xlabel("Range (km)")
    plt.ylabel("Capon power (norm.)")
    plt.title("High-resolution range spectrum (synthetic)")
    plt.tight_layout()
    plt.show()

    # Save as image for later use
    # plt.imsave("tmp/synthetic_ionogram.png", P, cmap="gray", vmin=0.0, vmax=1.0)
