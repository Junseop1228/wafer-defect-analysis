import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F


def plot_confusion_matrix(y_true, y_pred, labels: list, save_path: str = None):
    """Normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_shap_summary(shap_values, X, feature_names: list, class_name: str, save_path: str = None):
    """SHAP bar plot for one class."""
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Create DataFrame for easier sorting
    df_shap = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    })
    df_shap = df_shap.sort_values(by='Importance', ascending=True).tail(20)  # Top 20

    plt.figure(figsize=(10, 8))
    plt.barh(df_shap['Feature'], df_shap['Importance'], color='skyblue')
    plt.xlabel('mean(|SHAP value|)')
    plt.title(f'SHAP Summary - {class_name}')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_3way_comparison(metrics_csv_path: str, save_path: str = None):
    """Bar chart comparing ML / CNN / Hybrid across macro_f1, scratch, donut."""
    if not os.path.exists(metrics_csv_path):
        print(f"Metrics file not found: {metrics_csv_path}")
        return

    df = pd.read_csv(metrics_csv_path)
    # Expecting columns: Model, macro_F1, Scratch_recall, Donut_recall

    models = df['Model'].values
    metrics = ['macro_F1', 'Scratch_recall', 'Donut_recall']

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(12, 6))
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            vals = df[metric].values
            plt.bar(x + i*width - width, vals, width, label=metric)

    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.title('3-Way Model Comparison')
    plt.xticks(x, models)
    plt.legend()
    plt.ylim(0, 1.1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_gradcam(model, input_tensor, target_class: int, class_name: str, save_path: str = None):
    """Grad-CAM heatmap using manual backward hook (no external library)."""
    model.eval()

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Hook into the last convolutional block (conv3)
    target_layer = model.conv3
    handle_fw = target_layer.register_forward_hook(forward_hook)

    # PyTorch >= 1.8 uses register_full_backward_hook
    handle_bw = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    out = model(input_tensor)

    # Backward pass
    model.zero_grad()
    target_score = out[0, target_class]
    target_score.backward()

    # Remove hooks
    handle_fw.remove()
    handle_bw.remove()

    # Get activations and gradients
    acts = activations[0].squeeze().detach().cpu().numpy()
    grads = gradients[0].squeeze().detach().cpu().numpy()

    # Global Average Pooling on gradients
    weights = np.mean(grads, axis=(1, 2))

    # Weighted combination
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    # Apply ReLU
    cam = np.maximum(cam, 0)

    # Normalize
    if np.max(cam) != 0:
        cam = cam / np.max(cam)

    # Resize to original image size (64x64)
    # Use scipy or PIL to avoid cv2 dependency if possible, but F.interpolate is easier
    cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    orig_size = (input_tensor.shape[2], input_tensor.shape[3])
    cam_resized = F.interpolate(cam_tensor, size=orig_size, mode='bilinear', align_corners=False)
    cam_resized = cam_resized.squeeze().numpy()

    # Original image for overlay
    img = input_tensor.squeeze().cpu().numpy()
    if len(img.shape) == 3:
        img = img[0]  # Take first channel if it's 3-channel

    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap='gray')
    plt.imshow(cam_resized, cmap='jet', alpha=0.5)
    plt.title(f'Grad-CAM: {class_name}')
    plt.colorbar(label='Importance')
    plt.axis('off')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
