import sys, os
sys.path.insert(0, r'C:\Users\userPC\anaconda3\envs\wm811k\Lib\site-packages')
os.environ['PATH'] = r'C:\Users\userPC\anaconda3\envs\wm811k\Library\bin;' + os.environ.get('PATH', '')

import mlflow
import xgboost
import optuna

print("Imports successful!")
print("MLflow:", mlflow.__version__)
print("XGBoost:", xgboost.__version__)
print("Optuna:", optuna.__version__)
