"""
Unit tests for src/features.py and src/evaluate.py
No data file needed — uses small numpy fixtures.
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.features import extract_features

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_map():
    m = np.ones((26, 26), dtype=np.uint8)
    m[0,:]=0; m[-1,:]=0; m[:,0]=0; m[:,-1]=0
    return m

@pytest.fixture
def center_defect_map():
    m = np.ones((26, 26), dtype=np.uint8)
    m[0,:]=0; m[-1,:]=0; m[:,0]=0; m[:,-1]=0
    m[10:16, 10:16] = 2
    return m

@pytest.fixture
def scratch_map():
    m = np.ones((26, 26), dtype=np.uint8)
    m[0,:]=0; m[-1,:]=0; m[:,0]=0; m[:,-1]=0
    for i in range(1, 25):
        m[i, i] = 2
    return m


# ---------------------------------------------------------------------------
# Tests: extract_features (Phase 1)
# ---------------------------------------------------------------------------

def test_extract_features_returns_dict(center_defect_map):
    feats = extract_features(center_defect_map)
    assert isinstance(feats, dict)

def test_feature_count(center_defect_map):
    import yaml
    cfg = yaml.safe_load(open('config.yaml'))
    drop_list = cfg.get('features', {}).get('dropped_features', [])
    feats = extract_features(center_defect_map, drop_list=drop_list)
    assert len(feats) == 20, f"Expected 20 features, got {len(feats)}"

def test_no_nan_values(center_defect_map):
    feats = extract_features(center_defect_map)
    for k, v in feats.items():
        assert v is not None and not np.isnan(float(v)), f"Feature {k} is NaN/None"

def test_defect_ratio_zero_on_normal_map(normal_map):
    assert extract_features(normal_map)['defect_ratio'] == 0.0

def test_defect_ratio_positive_on_defect_map(center_defect_map):
    assert extract_features(center_defect_map)['defect_ratio'] > 0.0

def test_radial_cdf_monotonic(center_defect_map):
    feats = extract_features(center_defect_map)
    cdf = [feats[f'radial_cdf_{i}'] for i in range(10) if f'radial_cdf_{i}' in feats]
    for i in range(len(cdf)-1):
        assert cdf[i] <= cdf[i+1] + 1e-6

def test_compactness_scratch_vs_center(scratch_map, center_defect_map):
    assert extract_features(scratch_map)['compactness'] < extract_features(center_defect_map)['compactness']


# ---------------------------------------------------------------------------
# Tests: Gate functions (Phase 2 — implemented, skip removed)
# ---------------------------------------------------------------------------

def test_check_gate2_binary_pass():
    import yaml
    from src.evaluate import check_gate2_binary
    cfg = yaml.safe_load(open('config.yaml'))
    result = check_gate2_binary(recall=0.95, cfg=cfg)
    assert result['passed'] is True
    assert 'details' in result

def test_check_gate2_binary_fail():
    import yaml
    from src.evaluate import check_gate2_binary
    cfg = yaml.safe_load(open('config.yaml'))
    assert check_gate2_binary(recall=0.50, cfg=cfg)['passed'] is False

def test_check_gate2_full_pass():
    import yaml
    from src.evaluate import check_gate2_full
    cfg = yaml.safe_load(open('config.yaml'))
    assert check_gate2_full(scratch_recall=0.80, donut_recall=0.80, cfg=cfg)['passed'] is True

def test_check_gate2_full_fail_scratch():
    import yaml
    from src.evaluate import check_gate2_full
    cfg = yaml.safe_load(open('config.yaml'))
    assert check_gate2_full(scratch_recall=0.50, donut_recall=0.80, cfg=cfg)['passed'] is False

def test_check_gate3_pass():
    import yaml, numpy as np
    from src.evaluate import check_gate3
    cfg = yaml.safe_load(open('config.yaml'))
    class_names = ['Normal','Edge-Ring','Edge-Loc','Center','Loc','Scratch','Random','Donut']
    # Perfect predictions using string labels
    y_true = [class_names[i % 8] for i in range(80)]
    y_pred = y_true[:]
    result = check_gate3(y_true, y_pred, class_names=class_names, cfg=cfg)
    assert result['passed'] is True

def test_check_gate3_fail():
    import yaml, numpy as np
    from src.evaluate import check_gate3
    cfg = yaml.safe_load(open('config.yaml'))
    class_names = ['Normal','Edge-Ring','Edge-Loc','Center','Loc','Scratch','Random','Donut']
    # Edge-Ring always predicted as Edge-Loc using string labels
    y_true = ['Edge-Ring']*50 + ['Edge-Loc']*50 + [class_names[i % 8] for i in range(80)]
    y_pred = ['Edge-Loc']*50 + ['Edge-Loc']*50 + [class_names[i % 8] for i in range(80)]
    result = check_gate3(y_true, y_pred, class_names=class_names, cfg=cfg)
    assert result['passed'] is False
