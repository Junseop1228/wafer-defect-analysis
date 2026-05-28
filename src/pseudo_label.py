import pandas as pd
import numpy as np
import joblib
import torch
import mlflow
import os

from src.features import extract_features
from src.models import WaferCNN

def generate_pseudo_labels(cfg: dict) -> dict:
    """
    1. LSWMD.pkl에서 레이블 없는 샘플 추출 (failureType == '' or [])
    2. src/features.py extract_features()로 피처 추출 (chunk 10000장씩)
    3. results/hybrid_model.pkl 로드
    4. predict_proba → confidence = max(proba) ≥ 0.95 필터링
    5. data/features_pseudo.pkl 저장
    Returns: {'n_total': int, 'n_accepted': int, 'class_dist': dict}
    """
    df = pd.read_pickle(cfg['data']['raw_path'])
    df['failureType'] = df['failureType'].apply(lambda x: x[0][0] if len(x) > 0 else '')
    df_unlabeled = df[df['failureType'] == ''].copy()
    n_total = len(df_unlabeled)
    print(f"Total unlabeled samples: {n_total}")
    
    xgb_model = joblib.load('results/hybrid_model.pkl')
    class_names = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Normal', 'Random', 'Scratch']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model = WaferCNN(num_classes=8).to(device)
    cnn_model.load_state_dict(torch.load(cfg['data']['cnn_weights'], map_location=device))
    cnn_model.eval()
    
    chunk_size = 10000
    confidence_thresh = cfg['cycles'].get('pseudo_label_confidence', 0.95)
    features_to_drop = cfg['features'].get('dropped_features', [])
    
    accepted_rows = []
    accepted_cnn_embs = []
    raw_results = []
    
    for start_idx in range(0, n_total, chunk_size):
        end_idx = min(start_idx + chunk_size, n_total)
        chunk = df_unlabeled.iloc[start_idx:end_idx]
        
        from joblib import Parallel, delayed
        manual_feats_list = Parallel(n_jobs=-1, backend="loky")(
            delayed(extract_features)(wmap, drop_list=features_to_drop) for wmap in chunk['waferMap'].values
        )
        
        wmaps_3c = []
        for wmap in chunk['waferMap'].values:
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
            
        df_manual = pd.DataFrame(manual_feats_list)
        X_manual = df_manual.values
        
        wmaps_tensor = torch.tensor(np.array(wmaps_3c))
        embs_list = []
        with torch.no_grad():
            for i in range(0, len(wmaps_tensor), 256):
                batch = wmaps_tensor[i:i+256].to(device)
                batch_embs = cnn_model.extract_features(batch).cpu().numpy()
                embs_list.append(batch_embs)
        embs = np.vstack(embs_list)
            
        X_hybrid = np.hstack([X_manual, embs])
        
        probas = xgb_model.predict_proba(X_hybrid)
        max_probas = np.max(probas, axis=1)
        preds = np.argmax(probas, axis=1)
        
        for i, idx_in_chunk in enumerate(chunk.index):
            raw_results.append({
                'sample_index': idx_in_chunk,
                'predicted_class': class_names[preds[i]],
                'max_confidence': max_probas[i],
                'probabilities': probas[i]
            })
        
        mask = max_probas >= confidence_thresh
        accepted_indices = np.where(mask)[0]
        
        for idx in accepted_indices:
            row_dict = manual_feats_list[idx].copy()
            row_dict['failureType'] = class_names[preds[idx]]
            accepted_rows.append(row_dict)
            accepted_cnn_embs.append(embs[idx])
            
        print(f"Processed {end_idx}/{n_total} - Accepted so far: {len(accepted_rows)}")
        
    df_pseudo = pd.DataFrame(accepted_rows)
    df_pseudo.to_pickle('data/features_pseudo.pkl')
    np.save('data/cnn_embeddings_pseudo.npy', np.array(accepted_cnn_embs))
    
    pd.DataFrame(raw_results).to_pickle('data/pseudo_raw_results.pkl')
    
    class_dist = df_pseudo['failureType'].value_counts().to_dict() if len(df_pseudo) > 0 else {}
    
    with mlflow.start_run(run_name="stage4_pseudo_label"):
        mlflow.log_param("confidence_thresh", confidence_thresh)
        mlflow.log_metric("n_accepted", len(df_pseudo))
        for k, v in class_dist.items():
            mlflow.log_metric(f"pseudo_count_{k}", v)
            
    return {
        'n_total': n_total,
        'n_accepted': len(df_pseudo),
        'class_dist': class_dist
    }


def merge_pseudo_with_labeled(cfg: dict) -> str:
    """
    features_labeled_v2.pkl + features_pseudo.pkl 합치기
    → data/features_augmented.pkl 저장
    Returns: save_path
    """
    df_labeled = pd.read_pickle(cfg['data']['features_v2'])
    df_pseudo = pd.read_pickle('data/features_pseudo.pkl')
    
    df_augmented = pd.concat([df_labeled, df_pseudo], ignore_index=True)
    save_path = 'data/features_augmented.pkl'
    df_augmented.to_pickle(save_path)
    
    try:
        labeled_embs = np.load(cfg['data']['cnn_embeddings'])
        pseudo_embs = np.load('data/cnn_embeddings_pseudo.npy')
        augmented_embs = np.vstack([labeled_embs, pseudo_embs])
        np.save('data/cnn_embeddings_augmented.npy', augmented_embs)
    except Exception as e:
        print(f"Could not merge CNN embeddings: {e}")
    
    return save_path
