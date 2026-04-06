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
| [SBF Parser](digisonde/parsers/sbf.md) | Read binary SBF ionogram amplitude/phase blocks |
| [MMM Parser](digisonde/parsers/mmm.md) | Read binary ModMax MMM ionogram blocks |
| [Image Parser](digisonde/parsers/image.md) | Extract ionogram images (`IonogramImageExtractor`) |
| [SAO XML Datatypes](digisonde/datatypes/saoxmldatatypes.md) | Dataclasses for SAO XML records |
| [SBF Datatypes](digisonde/datatypes/sbfdatatypes.md) | Dataclasses for SBF binary format |
| [MMM Datatypes](digisonde/datatypes/mmmdatatypes.md) | Dataclasses for MMM/ModMax binary format |
| [DFT Datatypes](digisonde/datatypes/dftdatatypes.md) | Dataclasses for DFT spectral format |
| [RSF Datatypes](digisonde/datatypes/rsfdatatypes.md) | Dataclasses for RSF binary records |
| [Raw IQ Parse](digisonde/raw/raw_parse.md) | DPS4D raw IQ → ionogram pipeline (`process`, `IonogramResult`) |
| [IQ Reader](digisonde/raw/iq_reader.md) | One-second `.bin` IQ file reader (`IQStream`) |
| [Raw Plots](digisonde/raw/raw_plots.md) | Plotting helpers for raw IQ ionograms (`RawPlots`, `AFRLPlots`) |

### VIPIR

| Module | Purpose |
|--------|---------|
| [NGI Utils](vipir/ngi/utils.md) | Time-zone conversion, smoothing, color helpers |
| [NGI Source](vipir/ngi/source.md) | `Dataset` / `DataSource` for loading NGI NetCDF files |
| [NGI Scale](vipir/ngi/scale.md) | Auto-scaling, noise profiling, trace extraction |
| [NGI Plot Utils](vipir/ngi/plotlib.md) | Ionogram and profile visualisation helpers |
| [RIQ Echo Extractor](vipir/riq/echo.md) | Dynasonde-style seven-parameter echo extraction (`Echo`, `EchoExtractor`) |
| [RIQ Ionogram Filter](vipir/riq/parsers/filter.md) | Five-stage post-extraction echo filter: RFI, EP, multi-hop, DBSCAN, temporal coherence (`IonogramFilter`) |
| [RIQ Read](vipir/riq/parsers/read_riq.md) | Low-level RIQ binary reader and threshold routines |
| [RIQ Utils](vipir/riq/utils.md) | RIQ format utility functions |
| [SCT Datatypes](vipir/riq/datatypes/sct.md) | System configuration table dataclass |
| [PCT Datatypes](vipir/riq/datatypes/pct.md) | Pulse configuration table dataclass |

### VIPIR Analysis

| Module | Purpose |
|--------|---------|
| [Analysis Overview](vipir/analysis/index.md) | Physics analysis sub-package overview, pipeline diagram, all public imports |
| [Es Imaging — Package](vipir/analysis/es_imaging/index.md) | Package overview: algorithm, instrument parameters, singularity constraints |
| [Es Imaging — capon.py](vipir/analysis/es_imaging/capon.md) | `EsCaponImager` (single-cube Capon imager), `EsImagingResult` — full constructor, `fit()`, internal methods, plot API |
| [Es Imaging — aggregator.py](vipir/analysis/es_imaging/aggregator.md) | `RiqAggregator` — multi-file A+B+C; `load()`, `combine()`, `fit()`; Option A/B/C combining strategy |
| [Absorption](vipir/analysis/absorption.md) | `AbsorptionAnalyzer` → `LOFResult`, `DifferentialResult`, `TotalAbsorptionResult`, `AbsorptionProfileResult` |
| [True Height Inversion](vipir/analysis/inversion.md) | `TrueHeightInversion` → `EDPResult` — Titheridge lamination method; N(h) profile |
| [Polarization](vipir/analysis/polarization.md) | `PolarizationClassifier` → `PolarizationResult` — O/X mode labelling from PP chirality |
| [Spread-F](vipir/analysis/spread_f.md) | `SpreadFAnalyzer` → `SpreadFResult` — range/frequency/mixed spread-F detection |
| [Scaler](vipir/analysis/scaler.md) | `IonogramScaler` → `ScaledParameters` — foE, foF2, h′F, MUF, bootstrap uncertainty |
| [Irregularities](vipir/analysis/irregularities.md) | `IrregularityAnalyzer` → `IrregularityProfile` — EP structure function, spectral index α |
| [NeXtYZ](vipir/analysis/nextyz.md) | `NeXtYZInverter` → `NeXtYZResult`, `WedgePlane` — 3-D WSI + Hamiltonian ray tracing (Zabotin et al. 2006) |
