<!-- 
Author(s): Shibaji Chakraborty

Disclaimer:
pynasonde is under the MIT license found in the root directory LICENSE.md 
Everyone is permitted to copy and distribute verbatim copies of this license 
document.

This version of the MIT Public License incorporates the terms
and conditions of MIT General Public License.
-->

## ngi.scale — NoiseProfile and AutoScaler (manual doc)

This page provides a short, manual reference for the primary classes in
`pynasonde.vipir.ngi.scale`. The repository's documentation build may not
have all optional binary dependencies (for example `cv2`/OpenCV or SciPy)
available during import; the original project uses `mkdocstrings` to
autogenerate API docs, but importing `ngi.scale` can fail in that
environment. To avoid build-time import errors this file documents the
classes manually.

### NoiseProfile

- Purpose: represent a noise profile used by the AutoScaler. The default
    profile type is exponential and a small constant multiplier can be set
    at initialization.
- Key constructor arguments:
    - `type` (str): profile type, default `'exp'`.
    - `constant` (float): initial scaling constant, default `1.5`.
- Main method:
    - `get_exp_profile(x: np.ndarray, a0: float, b0: float, x0: float) -> np.ndarray`
        Returns an exponential profile computed as `a0 * exp(-b0 * x / x0)`.
::: pynasonde.vipir.ngi.scale.NoiseProfile
    handler: python
    options:
        show_root_heading: true
        show_source: false
        members:
            - get_exp_profile

::: pynasonde.vipir.ngi.scale.AutoScaler
    handler: python
    options:
        show_root_heading: true
        show_source: false
        members:
            - extract
            - to_binary_traces
            - fit_parabola
    - `to_binary_traces(...)` — threshold the image (Otsu + morphological operations), cluster candidate points with DBSCAN and fit parabolic curves to traces.
