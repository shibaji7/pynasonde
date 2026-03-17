# VIPIR Ionogram (pynasonde.vipir.ngi)

This section documents the NGI ionogram sub-package.

Contents

- `utils.md` — time-zone conversion, smoothing, colour helpers, gridded-parameter utilities
- `source.md` — `Dataset` dataclass and `DataSource` file manager for NGI NetCDF files
- `scale.md` — `AutoScaler`, `NoiseProfile`, median filter, image segmentation, parabola fitting
- `plotlib.md` — ionogram and electron-density-profile visualisation helpers

Example quick start

```py
from pynasonde.vipir.ngi.source import DataSource, Dataset

src = DataSource(source_folder="data/", file_ext="*.ngi.bz2")
ds = Dataset()
# ds.__initialize__(xr.open_dataset(src.file_names[0]))
```
