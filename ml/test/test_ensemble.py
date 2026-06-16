import pandas as pd
import joblib
from pathlib import Path


import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.append(str(root))
from ml.predictor.ensemble_predictor import backtest_ensemble
rolling_df = pd.read_csv(root / "data" / "rolling_df.csv")

models_folder = root / "ml"/ "models"

ridge_predictors = joblib.load(
    models_folder / "selected_predictors_ridge.pkl"
)

logistic_predictors = joblib.load(
    models_folder / "selected_predictors_logistic.pkl"
)

weights = [0.00, 0.02, 0.05, 0.10, 0.15]

for w in weights:
    acc = backtest_ensemble(
        rolling_df,
        ridge_predictors,
        logistic_predictors,
        ridge_weight=w
    )

    print(f"Weight={w:.2f} Accuracy={acc:.2%}")