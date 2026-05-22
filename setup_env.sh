# Environment Setup — WM-811K
# Run this once to create the conda environment

# Step 1: Create environment
conda create -n wm811k python=3.10 -y

# Step 2: Activate and install
conda activate wm811k
pip install -r requirements.txt

# Step 3: Verify
python -c "import torch, sklearn, mlflow, shap, optuna, streamlit; print('All packages OK')"

# Step 4: Register Jupyter kernel (for notebooks)
pip install ipykernel
python -m ipykernel install --user --name wm811k --display-name "Python (wm811k)"

# Step 5: Download data
# Place LSWMD.pkl in data/ manually (Kaggle download required)
# https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
