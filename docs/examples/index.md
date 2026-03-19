# Examples

<div class="hero">
  <h3>Executable Workflows</h3>
  <p>Hands-on examples for loading ionosonde data, processing soundings, and producing publication-ready figures.</p>
</div>

## DIGISONDE Examples

<div class="doc-card-grid">
  <div class="doc-card">
    <strong>DVL — Drift Velocity Stack Plot</strong>
    Load a full day of DPS4D <code>.DVL</code> files in parallel and produce a three-panel stacked drift velocity figure with a virtual-height overlay.
    <br><a href="digisonde/dvl/">Open Example</a>
  </div>
  <div class="doc-card">
    <strong>SKY — Sky Map Visualization</strong>
    Parse DIGISONDE <code>.SKY</code> files, build single-panel polar sky maps colored by Doppler frequency, and combine multiple soundings into a multi-panel comparison figure.
    <br><a href="digisonde/sky/">Open Example</a>
  </div>
  <div class="doc-card">
    <strong>SAO — Height Profiles and F2 Diagnostics</strong>
    Extract electron-density height profiles and scaled F2-layer parameters from DPS4D <code>.SAO</code> files; produce time–height and dual-axis line plots.
    <br><a href="digisonde/sao/">Open Example</a>
  </div>
  <div class="doc-card">
    <strong>SAO + DFT — Isodensity Contours, Doppler Waterfall, and Spectra</strong>
    Build a daily isodensity contour from hundreds of <code>.SAO</code> files, then visualize the Doppler waterfall and per-height spectra from a single <code>.DFT</code> file.
    <br><a href="digisonde/sao_dft/">Open Example</a>
  </div>
  <div class="doc-card">
    <strong>RSF — Direction-Coded Ionogram and Daily Directogram</strong>
    Parse raw DPS4D <code>.RSF</code> sounding files, render a direction-coded ionogram for a single record, and stack a full day into a directogram (time vs. ground distance).
    <br><a href="digisonde/rsf_direction_ionogram/">Open Example</a>
  </div>
  <div class="doc-card">
    <strong>RSF — Parse and Inspect Raw Sounding File</strong>
    Low-level walkthrough: load a single <code>.RSF</code> file, parse all blocks and frequency groups into structured Python dataclasses, and inspect headers programmatically.
    <br><a href="digisonde/rsf/">Open Example</a>
  </div>
</div>

## VIPIR Examples

<div class="doc-card-grid">
  <div class="doc-card">
    <strong>RIQ — Ionogram from Raw Capture</strong>
    Read a VIPIR <code>.RIQ</code> file, clean the ionogram with the adaptive gain filter, and plot O/X-mode power on a frequency–virtual-height canvas.
    <br><a href="vipir/proc_riq/">Open Example</a>
  </div>
  <div class="doc-card">
    <strong>NGI — Frequency–Time Interval (FTI) Plot</strong>
    Load a day of VIPIR NGI ionogram cubes in parallel, flatten per-band power grids into a long-form dataframe, and produce O-mode FTI stacked panels.
    <br><a href="vipir/fti/">Open Example</a>
  </div>
  <div class="doc-card">
    <strong>NGI — AutoScaler Sanity-Check Figures</strong>
    Stage a day of NGI files, run the full autoscaling pipeline (median filter → image segmentation → Otsu + DBSCAN binary traces), and emit a QA sanity-check figure.
    <br><a href="vipir/scale_module/">Open Example</a>
  </div>
</div>

## Figure Gallery

![DVL Stack Plot](figures/stackplots_dvl.png)
![Single Sky Map](figures/single_skymap.png)
![Sky Map Panels](figures/panel_skymaps.png)
![SAO Height Profile](figures/stack_sao_ne.png)
![SAO F2 Diagnostics](figures/stack_sao_F2.png)
![RSF Direction Ionogram](figures/rsf_direction_ionogram_KR835.png)
![RSF Daily Directogram](figures/rsf_directogram_KR835_daily.png)
![SAO Isodensity Contours](figures/sao_isodensity_KR835.png)
![DFT Doppler Waterfall](figures/dft_doppler_waterfall_KR835.png)
![DFT Doppler Spectra](figures/dft_doppler_spectra_KR835.png)
![VIPIR Ionogram from RIQ](figures/ionogram_from_riq.png)
![VIPIR FTI Interval Plot](figures/fti.WI937.2022j.png)
![NGI AutoScaler QA](figures/ngi.scaler.png)
