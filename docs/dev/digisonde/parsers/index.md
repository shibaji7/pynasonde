# Digisonde parsers

<span class="api-badge api-package">P</span>
`pynasonde.digisonde.parsers` — SAO, SKY, DFT, RSF, DVL, EDP, SBF, MMM, and image parsers.

This page links individual parser modules and provides brief usage examples.

- [SAO parser](sao.md)
- [SKY parser](sky.md)
- [Image parser](image.md)
- [DFT parser](dft.md)
- [RSF parser](rsf.md)
- [SBF parser](sbf.md)
- [MMM parser](mmm.md)
- [GRM splitter](grm.md)
- [EDP parser](edp.md)
- [DVL parser](dvl.md)


## Examples

```py
from pynasonde.digisonde.parsers.sky import SkyExtractor

ext = SkyExtractor('path/to/file.SKY', extract_time_from_name=True)
df = ext.extract().to_pandas()
```
