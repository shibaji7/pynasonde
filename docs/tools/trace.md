<!-- 
Author(s): Shibaji Chakraborty

Disclaimer:

-->
# Trace, finding the correct traces from the ionogram

### Overview
`trace.py` contains utilities to extract echo traces and compute phase/phase-velocity from radar IQ data. The key functions are:

- `compute_phase(i, q)` — simple phase wrapper.
- `get_clean_iq_by_heights(pulse_i, pulse_q, ...)` — trims data to a height/range window and computes per-sample power.
- `extract_echo_traces(sct, pulse_i, pulse_q, ...)` — high-level trace extraction that finds peak gates, forms clean ranges for frequency blocks, and returns indices of valid frequency blocks.
- `compute_phase_velocity(sct, pulse_i, pulse_q, ...)` — computes per-pulse I/Q at peak gates and derives phase arrays (used by `extract_echo_traces`).

All operations favour vectorized numpy operations (no Python loops for core indexing/aggregation).

---

## Function: compute_phase(i, q)

- Signature
  ```py
  def compute_phase(i, q) -> np.ndarray
  ```
- Purpose
  Compute phase (in radians), mapped to [0, 2π).
- Inputs
  - `i`: array_like — I components
  - `q`: array_like — Q components (same shape as `i`)
- Behavior
  Returns `np.arctan2(q, i) % (2 * np.pi)`.
- Notes
  Uses `np.arctan2` which handles all quadrants and zeros consistently.

---

## Function: get_clean_iq_by_heights

- Signature
  ```py
  def get_clean_iq_by_heights(
      pulse_i: np.ndarray,
      pulse_q: np.ndarray,
      f1_range_low: Optional[int] = 70,
      f1_range_high: Optional[int] = 1000,
      range_dev: Optional[float] = 1.5,
  ) -> (pulse_i_range, pulse_q_range, power, n_pulses, f1_rlow, f1_rhigh)
  ```
- Purpose
  Trim the full `pulse_i`/`pulse_q` arrays to a sub-range window (height gate indices derived from `f1_range_low`/`f1_range_high`) and compute the power per sample.
- Expected input shapes
  - `pulse_i`, `pulse_q`: (n_pulses, n_total_gates, n_rx_channels) — typically `n_rx_channels==2` (E-W and N-S).
- Outputs
  - `pulse_i_range`: sliced I data (n_pulses, n_selected_gates, n_rx_channels)
  - `pulse_q_range`: sliced Q data (n_pulses, n_selected_gates, n_rx_channels)
  - `power`: sqrt(i**2 + q**2) shape (n_pulses, n_selected_gates, n_rx_channels)
  - `n_pulses`: integer
  - `f1_rlow`, `f1_rhigh`: integer gate indices used for slicing
- Algorithm
  - Convert physical range bounds to gate indices using `range_dev` (gates per unit).
  - Slice the input arrays.
  - Compute per-sample magnitude `power = sqrt(I^2 + Q^2)`.

---

## Function: extract_echo_traces

- Signature (abridged)
  ```py
  def extract_echo_traces(
      sct: SctType,
      pulse_i: np.ndarray,
      pulse_q: np.ndarray,
      f1_range_low: Optional[int] = 70,
      f1_range_high: Optional[int] = 1000,
      snr_variance_threshold: Optional[float] = 1.1,
      f2max: Optional[float] = 13000,
      range_dev: Optional[float] = 1.5,
  ) -> np.ndarray
  ```
- Purpose
  Discover valid frequency blocks (good echoes) by:
  1. Finding the peak gate (per pulse) using SNR on receiver 0.
  2. Grouping pulses into frequency blocks (`sct.frequency.pulse_count` pulses per set).
  3. Computing variance of peak gates across block pulses; if variance < threshold, assign block mean as the clean range.
  4. Return indices of blocks that have valid frequencies and clean ranges within limits.
  5. Calls `compute_phase_velocity(...)` to compute phase arrays (side effect).
- Key details
  - Peak detection uses:
    ```py
    peak_gates = np.nanargmax(20 * np.log10(power[..., 0]), axis=1)
    ```
    i.e., SNR-like value on RX 0; result is shape `(n_pulses,)`.
  - `num_sets = int(n_pulses / sct.frequency.pulse_count)` partitions pulses into `num_sets`.
  - `range_blocks = peak_gates[: num_sets * pulse_count].reshape(num_sets, pulse_count)`
  - Variance used to decide if a block is "clean": `variances < snr_variance_threshold`.
  - `clean_range[mask] = means[mask]` — blocks with low variance use the mean peak gate as the block's range.
  - A `good_index` array of block indices is returned, filtered by frequency (`block_freq < f2max` and positive) and range bounds.
- Tune types
  - `tune_type == 1` and `tune_type >= 4` supported (they differ in how `block_freq` is chosen)
  - `tune_type == 2` and `tune_type == 3` are NotImplemented.
- Returns
  - `good_index`: 1-D numpy array of the indices of valid blocks.

---

## Function: compute_phase_velocity

- Signature (abridged)
  ```py
  def compute_phase_velocity(
      sct: SctType,
      pulse_i: np.ndarray,
      pulse_q: np.ndarray,
      ...
  ):
  ```
- Purpose
  Compute per-pulse I/Q at peak gates and derived phase arrays for further Doppler / O/X computations.
- Core steps
  1. Call `get_clean_iq_by_heights` to get `pulse_i_range`, `pulse_q_range`, `power`, `n_pulses`.
  2. Compute `peak_gates` using the same `np.nanargmax(20*log10(...), axis=1)`.
  3. Extract per-pulse I/Q at the selected gate:
     - Vectorized selection:
       ```py
       idx = np.arange(n_pulses)
       pulse_q_peakrange = pulse_q_range[idx, peak_gates, :]   # shape (n_pulses, 2)
       pulse_i_peakrange = pulse_i_range[idx, peak_gates, :]   # shape (n_pulses, 2)
       ```
  4. Compute `phase = np.unwrap(np.arctan2(pulse_q_peakrange, pulse_i_peakrange))`.
  5. (Commented) assign `xphases` / `yphases` per receiver channel from `phase[..., 0]`, `phase[..., 1]`.
- Outputs / side effects
  - Computes `pulse_*_peakrange` (n_pulses, 2) and `phase` (n_pulses, 2) for next-stage processing.
  - Currently returns `None` (prints shapes) — the function should be extended to return useful arrays or modify passed containers.

---

## Shapes summary (typical)

- pulse_i, pulse_q: (n_pulses, n_total_gates, n_rx) — commonly n_rx == 2 (E-W, N-S)
- pulse_i_range, pulse_q_range: (n_pulses, n_selected_gates, n_rx)
- power: (n_pulses, n_selected_gates, n_rx)
- peak_gates: (n_pulses,)  — gate indices per pulse
- pulse_i_peakrange, pulse_q_peakrange: (n_pulses, 2) — one 2-element vector per pulse (RX channels)
- phase: (n_pulses, 2)

---

## Algorithmic complexity
- Most heavy work is vectorized numpy ops:
  - power computation: O(n_pulses * n_gates * n_rx)
  - argmax across gates: O(n_pulses * n_gates)
  - grouping/variance: O(num_sets * pulse_count)
- Memory: holds sliced arrays of size (n_pulses * selected_gates * n_rx). Vectorized selection doesn't duplicate large arrays unnecessarily.

---

## Edge cases & recommended defensive checks
- log10 of zero: `20*np.log10(power[..., 0])` will produce `-inf` for zero power. Consider using `np.maximum(power[..., 0], eps)` with small eps (e.g., `1e-12`) to avoid `-inf` or use `np.nan` treatment if appropriate.
- `np.nanargmax` raises if an entire row is NaN — handle pulses with all-NaN power specially (e.g., mark `peak_gates` invalid).
- When pulses tie for maximum across multiple gates:
  - `np.argmax` returns the first occurrence.
  - If you want to average I/Q across all tied gates, compute a mask `lines == maxvals[:, None]` and average over those gates.
- `peak_gates` must be clipped before array indexing (`np.clip(peak_gates, 0, n_gates-1)`).
- `compute_phase_velocity` currently has no return value or exported arrays — it is called as side-effect; consider returning the computed phase arrays.

---

## Suggested improvements / TODOs
- Return `pulse_*_peakrange` and `phase` from `compute_phase_velocity` (instead of only printing).
- Replace `np.nanargmax` usage with `np.nanmax` + mask for more explicit handling of invalid pulses if necessary.
- Protect `log10` with a minimum epsilon (or treat zero-power rows via mask).
- Add unit tests that:
  - Validate shapes for synthetic random data.
  - Validate tie-breaking vs averaging behavior for tied peaks.
- Implement `tune_type==2` and `tune_type==3` or raise clearer guidance.

---

## Example usage

```py
# Given sct: SctType, pulse_i/pulse_q as described
good_blocks = extract_echo_traces(sct, pulse_i, pulse_q)
# To get phase arrays:
# Modify compute_phase_velocity to return phase arrays:
phase = compute_phase_velocity(sct, pulse_i, pulse_q)
# or use get_clean_iq_by_heights + manual selection:
pulse_i_range, pulse_q_range, power, n_pulses, rlow, rhigh = get_clean_iq_by_heights(pulse_i, pulse_q)
peak_gates = np.nanargmax(20*np.log10(np.maximum(power[...,0], 1e-12)), axis=1)
idx = np.arange(n_pulses)
pulse_q_peakrange = pulse_q_range[idx, np.clip(peak_gates, 0, pulse_q_range.shape[1]-1), :]
pulse_i_peakrange = pulse_i_range[idx, np.clip(peak_gates, 0, pulse_i_range.shape[1]-1), :]
phase = np.arctan2(pulse_q_peakrange, pulse_i_peakrange) % (2*np.pi)
```

---

If you want, I can:
- produce a ready-to-run `compute_phase_velocity` that returns `(pulse_i_peakrange, pulse_q_peakrange, phase)` and properly handles NaNs/ties; or
- write a short unit test (pytest) with synthetic data to verify shapes and tie-handling. Which would you like next?
