import numpy as np
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from skimage.transform import radon

# Stage 3 — Feature Engineering Core (Priority 1)
# Groups: Density / Radial CDF / Angular / Geometry / Morphology / Moran's I / DBSCAN

def extract_defect_coords(wafer_map: np.ndarray) -> np.ndarray:
    return np.argwhere(wafer_map == 2)

def calc_defect_ratio(wafer_map: np.ndarray) -> float:
    coords = extract_defect_coords(wafer_map)
    total_die = np.sum(wafer_map > 0)
    if total_die == 0:
        return 0.0
    return len(coords) / total_die

def calc_radial_cdf(wafer_map: np.ndarray, n_bins=10) -> list:
    coords = extract_defect_coords(wafer_map)
    if len(coords) == 0:
        return [0.0] * n_bins
    
    h, w = wafer_map.shape
    center_y, center_x = (h - 1) / 2.0, (w - 1) / 2.0
    max_radius = np.sqrt(center_y**2 + center_x**2)
    
    if max_radius == 0:
        return [0.0] * n_bins
        
    distances = np.sqrt((coords[:, 0] - center_y)**2 + (coords[:, 1] - center_x)**2)
    normalized_distances = distances / max_radius
    
    cdf = []
    for i in range(1, n_bins + 1):
        threshold = i / n_bins
        fraction = np.sum(normalized_distances <= threshold) / len(coords)
        cdf.append(float(fraction))
        
    return cdf

def calc_edge_sectors(wafer_map: np.ndarray) -> list:
    coords = extract_defect_coords(wafer_map)
    if len(coords) == 0:
        return [0.0] * 8
        
    h, w = wafer_map.shape
    center_y, center_x = (h - 1) / 2.0, (w - 1) / 2.0
    max_radius = np.sqrt(center_y**2 + center_x**2)
    
    if max_radius == 0:
        return [0.0] * 8
        
    distances = np.sqrt((coords[:, 0] - center_y)**2 + (coords[:, 1] - center_x)**2)
    angles = np.arctan2(coords[:, 0] - center_y, coords[:, 1] - center_x)
    
    # Filter for edge region (radius > 0.85 * max_radius)
    edge_mask = distances > 0.85 * max_radius
    edge_angles = angles[edge_mask]
    
    if len(edge_angles) == 0:
        return [0.0] * 8
        
    edge_angles_2pi = (edge_angles + 2 * np.pi) % (2 * np.pi)
    sector_bins = np.linspace(0, 2 * np.pi, 9)
    hist, _ = np.histogram(edge_angles_2pi, bins=sector_bins)
    
    # Normalize density in sector
    edge_sector = hist / len(coords) 
    
    return [float(x) for x in edge_sector]

def calc_angular_features(wafer_map: np.ndarray) -> float:
    edge_sector = calc_edge_sectors(wafer_map)
    return float(np.std(edge_sector))

def calc_geometry(wafer_map: np.ndarray) -> tuple:
    coords = extract_defect_coords(wafer_map)
    if len(coords) == 0:
        return 0.0, 0.0
        
    h, w = wafer_map.shape
    center_y, center_x = (h - 1) / 2.0, (w - 1) / 2.0
    
    cy = np.mean(coords[:, 0]) - center_y
    cx = np.mean(coords[:, 1]) - center_x
    
    norm_cy = cy / center_y if center_y > 0 else 0.0
    norm_cx = cx / center_x if center_x > 0 else 0.0
    
    return float(norm_cx), float(norm_cy)

def calc_morphology(wafer_map: np.ndarray) -> tuple:
    coords = extract_defect_coords(wafer_map)
    if len(coords) < 3:
        return 0.0, 0.0
        
    try:
        hull = ConvexHull(coords)
        area = hull.volume 
        perimeter = hull.area 
        compactness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0
    except Exception:
        compactness = 0.0
        
    min_y, min_x = np.min(coords, axis=0)
    max_y, max_x = np.max(coords, axis=0)
    
    height = max_y - min_y + 1
    width = max_x - min_x + 1
    
    aspect_ratio = float(width / height) if height > 0 else 0.0
    
    return float(compactness), float(aspect_ratio)

def calc_morans_i(wafer_map: np.ndarray) -> float:
    """Approximates Moran's I using a grid-based lattice for speed."""
    coords = extract_defect_coords(wafer_map)
    if len(coords) < 2:
        return 0.0
        
    h, w = wafer_map.shape
    grid_size_y = max(1, h // 8)
    grid_size_x = max(1, w // 8)
    
    grid_h = int(np.ceil(h / grid_size_y))
    grid_w = int(np.ceil(w / grid_size_x))
    
    grid = np.zeros((grid_h, grid_w))
    for y, x in coords:
        grid[y // grid_size_y, x // grid_size_x] += 1
        
    N = grid.size
    x_bar = np.mean(grid)
    
    var_term = np.sum((grid - x_bar)**2)
    if var_term == 0:
        return 0.0
        
    W = 0
    cov_term = 0
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    dev = grid - x_bar
    
    for dy, dx in shifts:
        y_start, y_end = max(0, -dy), grid_h - max(0, dy)
        x_start, x_end = max(0, -dx), grid_w - max(0, dx)
        
        if y_start >= y_end or x_start >= x_end:
            continue
            
        W += (y_end - y_start) * (x_end - x_start)
        cov_term += np.sum(dev[y_start:y_end, x_start:x_end] * dev[y_start+dy:y_end+dy, x_start+dx:x_end+dx])
        
    if W == 0:
        return 0.0
        
    morans_i = (N / W) * (cov_term / var_term)
    return float(morans_i)

def calc_n_clusters(wafer_map: np.ndarray) -> int:
    coords = extract_defect_coords(wafer_map)
    if len(coords) < 3:
        return 0
        
    db = DBSCAN(eps=3, min_samples=3)
    labels = db.fit_predict(coords)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return int(n_clusters)

def calc_lot_consistency(wafer_map: np.ndarray, lot_df=None) -> float:
    if lot_df is None or 'failureType' not in lot_df.columns or len(lot_df) == 0:
        return 0.5
    
    counts = lot_df['failureType'].value_counts()
    if len(counts) == 0:
        return 0.5
        
    return float(counts.iloc[0] / len(lot_df))

def calc_radon(wafer_map: np.ndarray) -> float:
    coords = extract_defect_coords(wafer_map)
    if len(coords) == 0:
        return 0.0
        
    binary_map = (wafer_map == 2).astype(float)
    sinogram = radon(binary_map, circle=False)
    
    return float(np.max(sinogram))

def extract_features(wafer_map: np.ndarray, lot_df=None) -> dict:
    """
    Extract engineered features from a single wafer map.
    Args:
        wafer_map: 2D array with values 0=outside, 1=normal, 2=defect
        lot_df: optional DataFrame with lotName and failureType columns
    Returns:
        dict of feature name -> float value
    """
    feats = {}
    
    feats['defect_ratio'] = calc_defect_ratio(wafer_map)
    
    cdf = calc_radial_cdf(wafer_map)
    for i in range(10):
        feats[f'radial_cdf_{i}'] = cdf[i]
        
    edge_sectors = calc_edge_sectors(wafer_map)
    for i in range(8):
        feats[f'edge_sector_{i}'] = edge_sectors[i]
        
    feats['edge_sector_std'] = float(np.std(edge_sectors))
    
    cx, cy = calc_geometry(wafer_map)
    feats['cx'] = cx
    feats['cy'] = cy
    
    compactness, aspect_ratio = calc_morphology(wafer_map)
    feats['compactness'] = compactness
    feats['aspect_ratio'] = aspect_ratio
    
    feats['morans_i'] = calc_morans_i(wafer_map)
    feats['n_clusters'] = calc_n_clusters(wafer_map)
    
    feats['lot_consistency'] = calc_lot_consistency(wafer_map, lot_df)
    feats['radon_max_value'] = calc_radon(wafer_map)
    
    return feats
