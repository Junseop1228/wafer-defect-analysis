import pandas as pd
import numpy as np
import yaml
import torch
import joblib
import os
from src.features import extract_features
from src.models import WaferCNN
from src.pseudo_label import merge_pseudo_with_labeled

def append_scratch():
    if not os.path.exists('data/pseudo_raw_results.pkl'):
        print("data/pseudo_raw_results.pkl not found!")
        return

    with open('config.yaml', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    raw_df = pd.read_pickle('data/pseudo_raw_results.pkl')
    
    scratch_thresh = cfg['cycles'].get('pseudo_scratch_confidence', 0.80)
    general_thresh = cfg['cycles'].get('pseudo_label_confidence', 0.95)
    
    scratch_samples = raw_df[(raw_df['predicted_class'] == 'Scratch') & 
                             (raw_df['max_confidence'] >= scratch_thresh) & 
                             (raw_df['max_confidence'] < general_thresh)]
                             
    if len(scratch_samples) == 0:
        print("No additional Scratch samples found in the 0.80~0.95 confidence range.")
        return
        
    print(f"Extracting features for {len(scratch_samples)} additional Scratch samples...")
    
    df_raw = pd.read_pickle(cfg['data']['raw_path'])
    df_raw['failureType'] = df_raw['failureType'].apply(lambda x: x[0][0] if len(x) > 0 else '')
    df_unlabeled = df_raw[df_raw['failureType'] == ''].copy()
    
    chunk = df_unlabeled.loc[scratch_samples['sample_index']]
    
    features_to_drop = cfg['features'].get('dropped_features', [])
    manual_feats_list = []
    wmaps_3c = []
    
    for wmap in chunk['waferMap'].values:
        feats = extract_features(wmap, drop_list=features_to_drop)
        manual_feats_list.append(feats)
        
        if wmap.shape != (64, 64):
            H, W = wmap.shape
            row_indices = np.floor(np.arange(64) * H / 64).astype(int)
            col_indices = np.floor(np.arange(64) * W / 64).astype(int)
            wmap_resized = wmap[row_indices, :][:, col_indices]
        else:
            wmap_resized = wmap
        wmap_3c = np.stack([wmap_resized]*3, axis=-1).astype(np.float32) / 2.0
        wmap_3c = np.transpose(wmap_3c, (2, 0, 1))
        wmaps_3c.append(wmap_3c)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model = WaferCNN(num_classes=8).to(device)
    cnn_model.load_state_dict(torch.load(cfg['data']['cnn_weights'], map_location=device))
    cnn_model.eval()
    
    wmaps_tensor = torch.tensor(np.array(wmaps_3c)).to(device)
    with torch.no_grad():
        embs = cnn_model.extract_features(wmaps_tensor).cpu().numpy()
        
    accepted_rows = []
    for i, idx in enumerate(chunk.index):
        row_dict = manual_feats_list[i].copy()
        row_dict['failureType'] = 'Scratch'
        accepted_rows.append(row_dict)
        
    df_new_pseudo = pd.DataFrame(accepted_rows)
    df_existing_pseudo = pd.read_pickle('data/features_pseudo.pkl')
    df_pseudo_combined = pd.concat([df_existing_pseudo, df_new_pseudo], ignore_index=True)
    df_pseudo_combined.to_pickle('data/features_pseudo.pkl')
    
    existing_embs = np.load('data/cnn_embeddings_pseudo.npy')
    combined_embs = np.vstack([existing_embs, embs])
    np.save('data/cnn_embeddings_pseudo.npy', combined_embs)
    
    print(f"Total Scratch pseudo labels: len(df_pseudo_combined[df_pseudo_combined['failureType'] == 'Scratch'])")
    
    print("Merging with labeled data...")
    save_path = merge_pseudo_with_labeled(cfg)
    print("Augmented data saved to:", save_path)
    print("Done!")

if __name__ == '__main__':
    append_scratch()
