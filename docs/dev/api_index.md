# API Documentation

<div class="hero" markdown>

**pynasonde** exposes a layered Python API for reading, parsing, and visualising ionosonde data from VIPIR and Digisonde instruments.  Use the navigation on the left to browse modules, or jump to a subsystem below.

</div>

## Symbol legend

Every heading in the reference pages carries a symbol that identifies what kind of Python object is documented.

<div class="doc-card-grid" markdown>

<div class="doc-card" markdown>
<span class="api-badge api-package">P</span> **Package / Module**

A Python package (directory with `__init__.py`) or a `.py` module file.
</div>

<div class="doc-card" markdown>
<span class="api-badge api-class">C</span> **Class**

A Python `class` definition, including dataclasses and abstract base classes.
</div>

<div class="doc-card" markdown>
<span class="api-badge api-method">M</span> **Method / Function**

An instance method, class method, static method, or module-level function.
</div>

<div class="doc-card" markdown>
<span class="api-badge" style="background:#f6eeff;color:#5a189a;border:1px solid #c77dff;">A</span> **Attribute / Property**

A class-level attribute, dataclass field, or `@property`.
</div>

</div>

---

## Subsystems

### Digisonde

| Module | Purpose |
|--------|---------|
| [Utils](digisonde/digi_utils.md) | Shared helpers: namespace conversion, station loading, grid utilities |
| [Plot Utils](digisonde/digi_plots.md) | Matplotlib wrappers for sky maps, drift velocity, EDP plots |
| [SAO Parser](digisonde/parsers/sao.md) | Read legacy fixed-width and XML SAO ionogram files |
| [SKY Parser](digisonde/parsers/sky.md) | Read SKY drift/direction files |
| [RSF Parser](digisonde/parsers/rsf.md) | Read raw RSF ionogram binary records |
| [DVL Parser](digisonde/parsers/dvl.md) | Read Digisonde Velocity/DVL files |
| [DFT Parser](digisonde/parsers/dft.md) | Read DFT spectral files |
| [EDP Parser](digisonde/parsers/edp.md) | Electron density profile extraction |
| [SAO XML Datatypes](digisonde/datatypes/saoxmldatatypes.md) | Dataclasses for SAO XML records |
| [SBF Datatypes](digisonde/datatypes/sbfdatatypes.md) | Dataclasses for SBF binary format |
| [DFT Datatypes](digisonde/datatypes/dftdatatypes.md) | Dataclasses for DFT spectral format |
| [RSF Datatypes](digisonde/datatypes/rsfdatatypes.md) | Dataclasses for RSF binary records |

### VIPIR

| Module | Purpose |
|--------|---------|
| [NGI Utils](vipir/ngi/utils.md) | Time-zone conversion, smoothing, color helpers |
| [NGI Source](vipir/ngi/source.md) | `Dataset` / `DataSource` for loading NGI NetCDF files |
| [NGI Scale](vipir/ngi/scale.md) | Auto-scaling, noise profiling, trace extraction |
| [NGI Plot Utils](vipir/ngi/plotlib.md) | Ionogram and profile visualisation helpers |
| [RIQ Trace](vipir/riq/parsers/trace.md) | IQ-data phase/velocity extraction from RIQ files |
| [RIQ Read](vipir/riq/parsers/read_riq.md) | Low-level RIQ binary reader and threshold routines |
| [RIQ Utils](vipir/riq/utils.md) | RIQ format utility functions |
| [SCT Datatypes](vipir/riq/datatypes/sct.md) | System configuration table dataclass |
| [PCT Datatypes](vipir/riq/datatypes/pct.md) | Pulse configuration table dataclass |
