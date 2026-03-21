# RIQ Parsers (pynasonde.vipir.riq.parsers)

This section documents the RIQ parser modules.

Contents

- `read_riq.md` — low-level RIQ binary reader, threshold detection, morphological noise removal, adaptive gain filter, `Pulset` container
- `filter.md` — `IonogramFilter`: five-stage post-extraction echo filter (RFI, EP, multi-hop, DBSCAN, temporal coherence)
