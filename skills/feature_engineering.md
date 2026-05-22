# Skill: Feature Engineering — WM-811K

> Antigravity reads this when implementing or modifying src/features.py
> All 28 features must follow these specs exactly.

---

## Implementation Rules

- Input to every function: `wafer_map` (2D numpy array, values 0/1/2), `lot_df` (optional, for lot-level features)
- Defect coordinates: `coords = np.argwhere(wafer_map == 2)`
- If `len(coords) == 0`: return feature vector of zeros — never raise exception
- All features normalized to [0, 1] range where applicable
- Single wafer processing time target: under 1 second for 99th percentile wafer size

---

## Feature Groups and Specs

### A. Density (1 feature)
- `defect_ratio`: `len(coords) / np.sum(wafer_map > 0)`

### B. Radial CDF (10 features)
- Compute distance from wafer center for each defect coord
- Normalize distances to [0, 1] by max possible radius
- `radial_cdf[i]`: fraction of defects within radius quantile i/10, for i in 0..9
- Key signal: Center → CDF rises fast at low radius; Edge-Ring → rises fast at high radius; Donut → S-curve

### C. Angular / Edge Sector (9 features)
- Divide edge region (radius > 0.85 * max_radius) into 8 sectors
- `edge_sector[i]`: defect density in sector i, for i in 0..7
- `edge_sector_std`: std of edge_sector[0..7]
- Key signal: Edge-Ring → uniform high std_low; Edge-Local → one sector spike

### D. Geometry (2 features)
- `cx`, `cy`: centroid of defect coords, normalized to [-1, 1] from wafer center
- Key signal: Center → (cx≈0, cy≈0); Edge-Local → one side offset

### E. Morphology (2 features)
- `compactness`: `4π * area / perimeter²` of defect cluster convex hull
- `aspect_ratio`: bounding box width / height
- Key signal: Scratch → low compactness, high aspect_ratio

### F. Spatial Autocorrelation (1 feature)
- `morans_i`: Moran's I statistic
- Use lattice-based weight matrix (queen contiguity on binned grid) — NOT full pairwise
- Grid resolution: min(wafer_map.shape) // 8 bins per axis
- Flag ①: This approximation trades precision for speed. Document as limitation.
- Key signal: Local → high positive I; Random → near-zero I

### G. Clustering (1 feature)
- `n_clusters`: number of DBSCAN clusters on defect coords
- params: `eps=3`, `min_samples=3` (pixel units, adjust if wafer is resized to 64×64)
- Key signal: Local → 1-3 clusters; Random → many clusters

### H. Lot-level (1 feature)
- `lot_consistency`: fraction of wafers in same lot with identical defect pattern class
- Requires lot_df with lotName column
- If lot_df is None → return 0.5 (neutral)
- Key signal: high → equipment-induced; low → random contamination

### I. Radon Transform (1 feature)
- `radon_max_value`: max value in Radon transform of binary defect map
- Use `skimage.transform.radon`
- Key signal: Scratch → very high radon_max_value due to linear alignment

---

## Function Signature

```python
def extract_features(wafer_map: np.ndarray, lot_df=None) -> dict:
    """
    Extract 28 engineered features from a single wafer map.
    Args:
        wafer_map: 2D array with values 0=outside, 1=normal, 2=defect
        lot_df: optional DataFrame with lotName and failureType columns
    Returns:
        dict of 28 feature name → float value
    """
```

---

## Priority Order

1. Must implement first: defect_ratio, radial_cdf[0..9], edge_sector_std, cx, cy, compactness, aspect_ratio, morans_i, n_clusters
2. Add next: lot_consistency, edge_sector[0..7]
3. Add last: radon_max_value
