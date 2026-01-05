import pandas as pd
import joblib
from predictor.ensemble_predictor import predict_game_ensemble_weighted
from pathlib import Path

# Load data
root = Path(__file__).resolve().parent
data_path = root / "data" / "rolling_df.csv"
rolling_df = pd.read_csv(data_path)

# Load models

models_folder = root / "models"

ridge_model = joblib.load(models_folder / "ridge_classifier_final.pkl")
ridge_predictors = joblib.load(models_folder / "selected_predictors_ridge.pkl")

logistic_model = joblib.load(models_folder / "logistic_model_final.pkl")
logistic_predictors = joblib.load(models_folder / "selected_predictors_logistic.pkl")

# Predict
result = predict_game_ensemble_weighted(
    rolling_df=rolling_df,
    ridge_model=ridge_model,
    ridge_predictors=ridge_predictors,
    logistic_model=logistic_model,
    logistic_predictors=logistic_predictors,
    home_team="NYK",
    away_team="DET",
    ridge_weight=0.02
)

print(f"Home team: {result['home_team']}")
print(f"Away team: {result['away_team']}")
print(f"Winner: {result['predicted_winner']}")
print(f"Loser: {result['predicted_loser']}")
print(f"Home win probability: {result['home_win_prob']:.2%}")
print(f"Away win probability: {result['away_win_prob']:.2%}")
