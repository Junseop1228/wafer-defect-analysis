import os
import pandas as pd
import numpy as np
import yaml
from collections import defaultdict

def main():
    print("Loading config and raw data...")
    cfg = yaml.safe_load(open('config.yaml', encoding='utf-8'))
    df = pd.read_pickle(cfg['data']['raw_path'])
    
    # Filter labeled only
    df['_ft_len'] = df['failureType'].apply(lambda x: len(x) if isinstance(x, np.ndarray) else 0)
    df = df[df['_ft_len'] > 0].copy()
    
    # Extract string
    if df['failureType'].dtype == object and len(df) > 0 and hasattr(df['failureType'].iloc[0], 'size'):
        df['_ft_str'] = df['failureType'].apply(lambda x: x[0][0] if hasattr(x, 'size') and x.size > 0 else '')
    else:
        df['_ft_str'] = df['failureType'].astype(str)
        
    merge_targets = cfg.get("classes", {}).get("merge_to_normal", [])
    df['_ft_str'] = df['_ft_str'].apply(lambda x: 'Normal' if x in merge_targets or x == 'none' else x)
    
    print("Aggregating lot-level timeseries data...")
    # Get all distinct lotNames (time-ordered by default since we assume index is time or lotName sorts roughly by time)
    lot_stats = df.groupby('lotName').agg(
        total=('waferIndex', 'count')
    ).reset_index()
    
    # Filter out small lots
    lot_stats = lot_stats[lot_stats['total'] >= 10].sort_values('lotName')
    
    classes = cfg.get("classes", {}).get("names", [])
    
    # For each class, calculate the defect rate per lot
    for cls in classes:
        cls_count = df[df['_ft_str'] == cls].groupby('lotName').size()
        lot_stats[cls] = lot_stats['lotName'].map(cls_count).fillna(0)
        lot_stats[f'{cls}_rate'] = lot_stats[cls] / lot_stats['total']
        
    # Also total defect rate
    df['is_defect'] = df['_ft_str'] != 'Normal'
    defect_count = df[df['is_defect']].groupby('lotName').size()
    lot_stats['Total_Defect'] = lot_stats['lotName'].map(defect_count).fillna(0)
    lot_stats['Total_Defect_rate'] = lot_stats['Total_Defect'] / lot_stats['total']
    
    # Mock AE scores per lot (average AE score or max AE score could be used in reality, 
    # but we don't have AE scores stored for all wafers in a dataframe yet).
    # For visualization, we will generate a mock anomaly score correlated with Total_Defect_rate
    rng = np.random.default_rng(42)
    lot_stats['AE_anomaly_score'] = lot_stats['Total_Defect_rate'] * 0.5 + rng.normal(0, 0.05, len(lot_stats))
    lot_stats['AE_anomaly_score'] = lot_stats['AE_anomaly_score'].clip(lower=0.0)
    
    out_path = 'data/spc_timeseries.csv'
    lot_stats.to_csv(out_path, index=False)
    print(f"Saved SPC timeseries data to {out_path} ({len(lot_stats)} lots).")

if __name__ == '__main__':
    main()
