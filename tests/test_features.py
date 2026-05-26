"""
Unit tests for src/features.py and src/evaluate.py
No data file needed — uses small numpy fixtures.

Marking policy:
  - Tests for IMPLEMENTED functions: no mark (must pass in CI)
  - Tests for PLANNED functions (not yet implemented): @pytest.mark.skip
    Remove skip marks as each function gets implemented in the corresponding Task.
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features import extract_features

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_map():
    m = np.ones((26, 26), dtype=np.uint8)
    m[0, :] = 0; m[-1, :] = 0; m[:, 0] = 0; m[:, -1] = 0
    return m

@pytest.fixture
def center_defect_map():
    m = np.ones((26, 26), dtype=np.uint8)
    m[0, :] = 0; m[-1, :] = 0; m[:, 0] = 0; m[:, -1] = 0
    m[10:16, 10:16] = 2
    return m

@pytest.fixture
def edge_defect_map():
    m = np.ones((26, 26), dtype=np.uint8)
    m[0, :] = 0; m[-1, :] = 0; m[:, 0] = 0; m[:, -1] = 0
    m[1:3, 1:-1] = 2
    m[-3:-1, 1:-1] = 2
    m[1:-1, 1:3] = 2
    m[1:-1, -3:-1] = 2
    return m

@pytest.fixture
def scratch_map():
    m = np.ones((26, 26), dtype=np.uint8)
    m[0, :] = 0; m[-1, :] = 0; m[:, 0] = 0; m[:, -1] = 0
    for i in range(1, 25):
        m[i, i] = 2
    return m


# ---------------------------------------------------------------------------
# Tests: extract_features (IMPLEMENTED — Phase 1 complete)
# ---------------------------------------------------------------------------

def test_extract_features_returns_dict(center_defect_map):
    feats = extract_features(center_defect_map)
    assert isinstance(feats, dict), "extract_features must return a dict"

def test_feature_count(center_defect_map):
    """With dropped_features applied, should return 20 features."""
    import yaml
    cfg = yaml.safe_load(open('config.yaml'))
    drop_list = cfg.get('features', {}).get('dropped_features', [])
    feats = extract_features(center_defect_map, drop_list=drop_list)
    assert len(feats) == 20, f"Expected 20 features, got {len(feats)}"

def test_no_nan_values(center_defect_map):
    feats = extract_features(center_defect_map)
    for k, v in feats.items():
        assert v is not None, f"Feature {k} is None"
        assert not np.isnan(float(v)), f"Feature {k} is NaN"

def test_defect_ratio_zero_on_normal_map(normal_map):
    feats = extract_features(normal_map)
    assert feats['defect_ratio'] == 0.0

def test_defect_ratio_positive_on_defect_map(center_defect_map):
    feats = extract_features(center_defect_map)
    assert feats['defect_ratio'] > 0.0

def test_radial_cdf_monotonic(center_defect_map):
    feats = extract_features(center_defect_map)
    cdf_vals = [feats[f'radial_cdf_{i}'] for i in range(10) if f'radial_cdf_{i}' in feats]
    if len(cdf_vals) >= 2:
        for i in range(len(cdf_vals) - 1):
            assert cdf_vals[i] <= cdf_vals[i+1] + 1e-6

def test_compactness_scratch_vs_center(scratch_map, center_defect_map):
    f_scratch = extract_features(scratch_map)
    f_center = extract_features(center_defect_map)
    assert f_scratch['compactness'] < f_center['compactness']


# ---------------------------------------------------------------------------
# Tests: Gate functions (PLANNED — skip until implemented)
# Remove skip marks as each Task completes.
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="check_gate2_binary not yet implemented — Phase 2 Task 1")
def test_check_gate2_binary_pass():
    import yaml
    from src.evaluate import check_gate2_binary
    cfg = yaml.safe_load(open('config.yaml'))
    result = check_gate2_binary(recall=0.95, cfg=cfg)
    assert 'passed' in result
    assert 'details' in result
    assert result['passed'] is True

@pytest.mark.skip(reason="check_gate2_binary not yet implemented — Phase 2 Task 1")
def test_check_gate2_binary_fail():
    import yaml
    from src.evaluate import check_gate2_binary
    cfg = yaml.safe_load(open('config.yaml'))
    result = check_gate2_binary(recall=0.50, cfg=cfg)
    assert result['passed'] is False

@pytest.mark.skip(reason="check_gate2_full not yet implemented — Phase 2 Task 6")
def test_check_gate2_full_pass():
    import yaml
    from src.evaluate import check_gate2_full
    cfg = yaml.safe_load(open('config.yaml'))
    result = check_gate2_full(scratch_recall=0.80, donut_recall=0.80, cfg=cfg)
    assert result['passed'] is True

@pytest.mark.skip(reason="check_gate3 not yet implemented — Phase 2 Task 8")
def test_check_gate3_pass():
    import yaml, numpy as np
    from src.evaluate import check_gate3
    cfg = yaml.safe_load(open('config.yaml'))
    dummy_cm = np.eye(7, dtype=int) * 100
    result = check_gate3(confusion_matrix=dummy_cm, class_names=['A','B','C','D','E','F','G'], cfg=cfg)
    assert result['passed'] is True
