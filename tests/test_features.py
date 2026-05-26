"""
Unit tests for src/features.py
No data file needed — uses small numpy fixtures.
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features import extract_features

# --- Fixtures ---

@pytest.fixture
def normal_map():
    """Wafer with no defects (all 1s inside, 0 outside)."""
    m = np.ones((26, 26), dtype=np.uint8)
    m[0, :] = 0; m[-1, :] = 0; m[:, 0] = 0; m[:, -1] = 0
    return m

@pytest.fixture
def center_defect_map():
    """Wafer with center cluster defect."""
    m = np.ones((26, 26), dtype=np.uint8)
    m[0, :] = 0; m[-1, :] = 0; m[:, 0] = 0; m[:, -1] = 0
    m[10:16, 10:16] = 2  # center cluster
    return m

@pytest.fixture
def edge_defect_map():
    """Wafer with edge-ring defect pattern."""
    m = np.ones((26, 26), dtype=np.uint8)
    m[0, :] = 0; m[-1, :] = 0; m[:, 0] = 0; m[:, -1] = 0
    m[1:3, 1:-1] = 2   # top edge
    m[-3:-1, 1:-1] = 2  # bottom edge
    m[1:-1, 1:3] = 2   # left edge
    m[1:-1, -3:-1] = 2  # right edge
    return m

@pytest.fixture
def scratch_map():
    """Wafer with linear scratch defect."""
    m = np.ones((26, 26), dtype=np.uint8)
    m[0, :] = 0; m[-1, :] = 0; m[:, 0] = 0; m[:, -1] = 0
    for i in range(1, 25):
        m[i, i] = 2  # diagonal line
    return m


# --- Tests: output shape & types ---

def test_extract_features_returns_dict(center_defect_map):
    feats = extract_features(center_defect_map)
    assert isinstance(feats, dict), "extract_features must return a dict"

def test_feature_count(center_defect_map):
    """After dropping 7 weak features, should return 21."""
    import yaml
    cfg = yaml.safe_load(open('config.yaml'))
    drop_list = cfg.get('features', {}).get('dropped_features', [])
    feats = extract_features(center_defect_map, drop_list=drop_list)
    assert len(feats) == 21, f"Expected 21 features, got {len(feats)}"

def test_no_nan_values(center_defect_map):
    feats = extract_features(center_defect_map)
    for k, v in feats.items():
        assert v is not None, f"Feature {k} is None"
        assert not np.isnan(float(v)), f"Feature {k} is NaN"

def test_defect_ratio_zero_on_normal_map(normal_map):
    feats = extract_features(normal_map)
    assert feats['defect_ratio'] == 0.0, "Normal map should have defect_ratio=0"

def test_defect_ratio_positive_on_defect_map(center_defect_map):
    feats = extract_features(center_defect_map)
    assert feats['defect_ratio'] > 0.0, "Defect map should have defect_ratio>0"

def test_radial_cdf_monotonic(center_defect_map):
    """Radial CDF bins should be non-decreasing."""
    feats = extract_features(center_defect_map)
    cdf_vals = [feats[f'radial_cdf_{i}'] for i in range(10) if f'radial_cdf_{i}' in feats]
    if len(cdf_vals) >= 2:
        for i in range(len(cdf_vals) - 1):
            assert cdf_vals[i] <= cdf_vals[i+1] + 1e-6, \
                f"Radial CDF not monotonic at bin {i}"

def test_compactness_scratch_vs_center(scratch_map, center_defect_map):
    """Scratch should have lower compactness (more elongated) than center cluster."""
    f_scratch = extract_features(scratch_map)
    f_center = extract_features(center_defect_map)
    # compactness: 4π*area/perimeter² — circle=1, line≈0
    assert f_scratch['compactness'] < f_center['compactness'], \
        "Scratch compactness should be lower than center cluster"


# --- Tests: Gate function signatures ---

def test_check_gate2_binary_signature():
    """check_gate2_binary must return dict with 'passed' and 'details' keys."""
    import yaml
    from src.evaluate import check_gate2_binary
    cfg = yaml.safe_load(open('config.yaml'))
    result = check_gate2_binary(recall=0.95, cfg=cfg)
    assert 'passed' in result
    assert 'details' in result
    assert result['passed'] is True

def test_check_gate2_binary_fail():
    import yaml
    from src.evaluate import check_gate2_binary
    cfg = yaml.safe_load(open('config.yaml'))
    result = check_gate2_binary(recall=0.50, cfg=cfg)
    assert result['passed'] is False
