import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class WaferMapDataset(Dataset):
    def __init__(self, wafer_maps, labels, transform=None):
        self.wafer_maps = wafer_maps
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.wafer_maps)
        
    def __getitem__(self, idx):
        wmap = self.wafer_maps[idx]
        
        # Resize to 64x64 with nearest neighbor to preserve integers (0,1,2)
        if wmap.shape != (64, 64):
            H, W = wmap.shape
            row_indices = np.floor(np.arange(64) * H / 64).astype(int)
            col_indices = np.floor(np.arange(64) * W / 64).astype(int)
            wmap = wmap[row_indices, :][:, col_indices]
            
        # Replicate to 3 channels (H, W, C)
        wmap_3c = np.stack([wmap]*3, axis=-1)
        
        # Normalize 0~2 to 0~1
        wmap_3c = wmap_3c.astype(np.float32) / 2.0
        
        # Transpose to (C, H, W)
        wmap_3c = np.transpose(wmap_3c, (2, 0, 1))
        
        if self.transform:
            wmap_3c = self.transform(wmap_3c)
            
        return torch.tensor(wmap_3c), torch.tensor(self.labels[idx], dtype=torch.long)

class WaferCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(WaferCNN, self).__init__()
        
        # Conv2d(3->32, k=3, p=1) -> BN -> ReLU -> MaxPool(2) # 64->32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Conv2d(32->64, k=3, p=1) -> BN -> ReLU -> MaxPool(2) # 32->16
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Conv2d(64->128, k=3, p=1) -> BN -> ReLU -> MaxPool(2) # 16->8
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1) # 8->1
        
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def extract_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return x
        
    def forward(self, x):
        x = self.extract_features(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
