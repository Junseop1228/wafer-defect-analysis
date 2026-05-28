import joblib, numpy as np, pandas as pd, yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
df = pd.read_pickle("data/features_labeled_v2.pkl")
df["failureType"] = df["failureType"].replace({"Near-full": "Normal"})
embs = np.load("data/cnn_embeddings.npy")
X = np.hstack([df[[c for c in df.columns if c != "failureType"]].values, embs])
y = df["failureType"].values
_, X_te, _, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = joblib.load("results/hybrid_model.pkl")

# If model outputs integers, we should decode or encode. But let's run exactly as requested or close to it.
try:
    preds = model.predict(X_te)
    classes = sorted(set(y))
    recalls = dict(zip(classes, recall_score(y_te, preds, average=None, labels=classes)))
    print(f"Scratch: {recalls.get('Scratch', 0):.4f}")
    print(f"Macro F1: {f1_score(y_te, preds, average='macro'):.4f}")
except Exception as e:
    print(f"Error evaluating: {e}")
    # Fix for integer output
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    _, _, _, y_te_enc = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
    preds = model.predict(X_te)
    classes_enc = sorted(set(y_te_enc))
    recalls = dict(zip(le.classes_, recall_score(y_te_enc, preds, average=None, labels=classes_enc)))
    print("Fixed integer prediction eval:")
    print(f"Scratch: {recalls.get('Scratch', 0):.4f}")
    print(f"Macro F1: {f1_score(y_te_enc, preds, average='macro'):.4f}")
