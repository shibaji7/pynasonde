<!-- 
Author(s): Shibaji Chakraborty

Disclaimer:

-->
# AutoScale-ISCA: _Automatic Ionogram Scaler using Image Segmentation and Clustering Analysis_
`AutoScale-ISCA` is a powerful tool within Pynasonde that automatically scales ionograms by leveraging advanced machine learning, image segmentation techniques, and clustering analysis. It provides a seamless, accurate way to identify and extract key ionospheric parameters, removing the need for manual scaling and enhancing data processing efficiency. Ideal for researchers and space weather analysts, `AutoScale-ISCA` ensures reliable and precise ionospheric data interpretation.

<figure markdown>
![Figure 01](../figures/ISCA-001.png)
<figcaption>Figure 01: Output from different stages of AutoScale-ISCA from a vertical ionogram. The event was at 15:50:03 UT on 04 April 2024. This was related to 2024 Great American Eclipse (GAE): (a) raw verical ionogram, (b) filtered ionogram (based on frequency-noise profiler and expected vertual height), (c) segmented using using fuzzy K-means, (d) converted to binary image of only segmenetd part, (e) identify the traces using density-based clustering (DBSCAN), and (f) estimated $fo_s$ and $hm_s$.
</figcaption>
</figure>
