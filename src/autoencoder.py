# Stage 7 — Convolutional Autoencoder
# Trained on unlabeled normal wafers for anomaly scoring

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
import yaml


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WaferMapAEDataset(Dataset):
    """Single-channel 64x64 wafer maps for autoencoder training.

    Input encoding: Binary mask
      Defect die (value 2) = 1.0
      Normal/Outside (value 0, 1) = 0.0
    """

    def __init__(self, wafer_maps: np.ndarray):
        self.wafer_maps = wafer_maps  # (N, 64, 64) pre-processed float32

    def __len__(self):
        return len(self.wafer_maps)

    def __getitem__(self, idx):
        return torch.tensor(self.wafer_maps[idx]).unsqueeze(0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class WaferAutoencoder(nn.Module):
    """Convolutional AE for wafer map anomaly detection.

    Encoder: Conv(1->16) -> Conv(16->32) -> Conv(32->64) -> Flatten -> FC(->128)
    Decoder: FC(128->) -> Reshape -> ConvT(64->32) -> ConvT(32->16) -> ConvT(16->1)
    Input/Output: 64x64x1 binary mask
    """

    def __init__(self, bottleneck: int = 128):
        super().__init__()

        # Encoder
        self.encoder_conv = nn.Sequential(
            # 64 -> 32
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 32 -> 16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 16 -> 8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # After pooling: 64 channels x 8 x 8 = 4096
        self.encoder_fc = nn.Linear(64 * 8 * 8, bottleneck)

        # Decoder
        self.decoder_fc = nn.Linear(bottleneck, 64 * 8 * 8)
        self.decoder_conv = nn.Sequential(
            # 8 -> 16
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 16 -> 32
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 32 -> 64
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid(),   # output in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        return self.encoder_fc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 64, 8, 8)
        return self.decoder_conv(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _load_unlabeled_wafer_maps(cfg: dict) -> np.ndarray:
    """Load unlabeled wafer maps from LSWMD.pkl (chunk-safe).

    In LSWMD.pkl, failureType is always np.ndarray:
      - Unlabeled: array([], shape=(0,0)) — size == 0
      - Labeled:   array([['none']]) or array([['Scratch']]) etc.
    """
    import pandas as pd

    raw_path = cfg["data"]["raw_path"]
    print(f"[AE] Loading raw data from {raw_path} ...")
    df = pd.read_pickle(raw_path)

    # Convert ndarray failureType to string ('' for unlabeled, name for labeled)
    df["_ft_str"] = df["failureType"].apply(
        lambda x: x[0][0] if hasattr(x, "size") and x.size > 0 else ""
    )
    df_unlabeled = df[df["_ft_str"] == ""].reset_index(drop=True)
    print(f"[AE] Unlabeled samples: {len(df_unlabeled):,}")

    wafer_maps = np.array(df_unlabeled["waferMap"].tolist(), dtype=object)
    return wafer_maps



def train_autoencoder(cfg: dict) -> dict:
    """Train WaferAutoencoder on unlabeled wafer maps.

    Steps:
    1. Load unlabeled wafer maps from LSWMD.pkl
    2. Convert to binary defect mask (64x64x1)
    3. Train AE with MSE loss for cfg epochs
    4. Save weights to results/ae_weights.pth
    5. Compute mean reconstruction error + 95th-percentile threshold

    Returns:
        dict with keys: model, mean_recon_error, threshold
    """
    ae_cfg = cfg.get("autoencoder", {})
    epochs = ae_cfg.get("epochs", 50)
    batch_size = ae_cfg.get("batch_size", 128)
    lr = ae_cfg.get("lr", 1e-3)
    bottleneck = ae_cfg.get("bottleneck", 128)
    max_samples = ae_cfg.get("max_samples", 50000)  # RAM safety cap
    weights_path = ae_cfg.get("weights_path", "results/ae_weights.pth")
    seed = cfg.get("seed", 42)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AE] Device: {device}")

    # Load data
    wafer_maps = _load_unlabeled_wafer_maps(cfg)

    # RAM safety & Diversity sampling: 
    # Calculate defect ratio to ensure we include wafers with > 0.01 defect ratio
    print("[AE] Calculating defect ratios for sampling diversity...")
    ratios = []
    for wmap in wafer_maps:
        wmap = np.array(wmap)
        defects = np.sum(wmap == 2)
        total = np.sum(wmap > 0)
        ratios.append(defects / total if total > 0 else 0)
    ratios = np.array(ratios)
    
    idx_high = np.where(ratios > 0.01)[0]
    idx_low = np.where(ratios <= 0.01)[0]
    
    rng = np.random.default_rng(seed)
    # We want up to 20% high defect ratio wafers in our 50k sample if available
    n_high_target = min(int(max_samples * 0.2), len(idx_high))
    n_low_target = min(max_samples - n_high_target, len(idx_low))
    
    idx_high_sampled = rng.choice(idx_high, n_high_target, replace=False) if n_high_target > 0 else []
    idx_low_sampled = rng.choice(idx_low, n_low_target, replace=False) if n_low_target > 0 else []
    
    sampled_indices = np.concatenate([idx_high_sampled, idx_low_sampled]).astype(int)
    rng.shuffle(sampled_indices)
    wafer_maps = wafer_maps[sampled_indices]
    print(f"[AE] Sampled {len(sampled_indices):,} wafers ({n_high_target:,} with defect_ratio > 0.01)")

    # Preprocess: resize to 64x64, convert to binary mask
    processed = []
    for wmap in wafer_maps:
        wmap = np.array(wmap, dtype=np.float32)
        if wmap.shape != (64, 64):
            H, W = wmap.shape
            row_idx = np.floor(np.arange(64) * H / 64).astype(int)
            col_idx = np.floor(np.arange(64) * W / 64).astype(int)
            wmap = wmap[row_idx, :][:, col_idx]
        processed.append((wmap == 2).astype(np.float32))

    wafer_maps_proc = np.stack(processed, axis=0)  # (N, 64, 64)
    print(f"[AE] Training samples: {len(wafer_maps_proc):,}")

    dataset = WaferMapAEDataset(wafer_maps_proc)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=0, pin_memory=(device.type == "cuda"))

    model = WaferAutoencoder(bottleneck=bottleneck).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = ae_cfg.get("pos_weight", 20.0)

    def weighted_mse_loss(input, target):
        weights = torch.where(target > 0.5, pos_weight, 1.0)
        return torch.mean(weights * (input - target) ** 2)

    experiment_name = cfg["mlflow"]["experiment_name"]
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="stage7_autoencoder"):
        mlflow.log_params({
            "ae_epochs": epochs,
            "ae_batch_size": batch_size,
            "ae_lr": lr,
            "ae_pos_weight": pos_weight,
            "ae_bottleneck": bottleneck,
            "ae_train_samples": len(wafer_maps_proc),
        })

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                x = batch.to(device)
                recon = model(x)
                loss = weighted_mse_loss(recon, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * x.size(0)
            avg_loss = total_loss / len(dataset)
            if (epoch + 1) % 10 == 0:
                print(f"[AE] Epoch [{epoch+1}/{epochs}] loss={avg_loss:.6f}")

        # Compute reconstruction errors on full training set for threshold
        model.eval()
        recon_errors = []
        eval_loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch in eval_loader:
                x = batch.to(device)
                recon = model(x)
                err = weighted_mse_loss(recon, x)
                # We need per-sample error for percentiles
                weights = torch.where(x > 0.5, pos_weight, 1.0)
                per_sample_err = torch.mean(weights * (recon - x) ** 2, dim=(1,2,3))
                recon_errors.extend(per_sample_err.cpu().numpy().tolist())

        recon_errors = np.array(recon_errors)
        mean_err = float(np.mean(recon_errors))
        threshold = float(np.percentile(recon_errors, 95))

        print(f"[AE] Mean recon error (normal): {mean_err:.6f}")
        print(f"[AE] Threshold (95th pct):      {threshold:.6f}")

        mlflow.log_metrics({
            "ae_mean_recon_error": mean_err,
            "ae_threshold": threshold,
        })

        # Save weights
        torch.save(model.state_dict(), weights_path)
        mlflow.log_artifact(weights_path)
        print(f"[AE] Weights saved to {weights_path}")

    return {"model": model, "mean_recon_error": mean_err, "threshold": threshold}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def compute_anomaly_scores(cfg: dict, wafer_maps: np.ndarray) -> np.ndarray:
    """Compute per-wafer reconstruction error (anomaly score).

    Args:
        cfg:        config.yaml dict
        wafer_maps: (N,) object array of variable-size wafer maps,
                    OR (N, 64, 64) float32 array already resized.

    Returns:
        anomaly_scores: shape (N,) float32
    """
    ae_cfg = cfg.get("autoencoder", {})
    bottleneck = ae_cfg.get("bottleneck", 128)
    batch_size = ae_cfg.get("batch_size", 128)
    weights_path = ae_cfg.get("weights_path", "results/ae_weights.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WaferAutoencoder(bottleneck=bottleneck).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Preprocess: resize to 64x64, convert to binary mask
    processed = []
    for wmap in wafer_maps:
        wmap = np.array(wmap, dtype=np.float32)
        if wmap.shape != (64, 64):
            H, W = wmap.shape
            row_idx = np.floor(np.arange(64) * H / 64).astype(int)
            col_idx = np.floor(np.arange(64) * W / 64).astype(int)
            wmap = wmap[row_idx, :][:, col_idx]
        processed.append((wmap == 2).astype(np.float32))

    wafer_maps_proc = np.stack(processed, axis=0)  # (N, 64, 64)
    dataset = WaferMapAEDataset(wafer_maps_proc)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)

    pos_weight = ae_cfg.get("pos_weight", 20.0)

    scores = []
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            recon = model(x)
            weights = torch.where(x > 0.5, pos_weight, 1.0)
            per_sample_err = torch.mean(weights * (recon - x) ** 2, dim=(1,2,3))
            scores.extend(per_sample_err.cpu().numpy().tolist())

    return np.array(scores, dtype=np.float32)
