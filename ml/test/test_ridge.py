import pandas as pd
import joblib
import sys
from pathlib import Path

# --- Set project root (one level above test/) ---
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from train.train_nba_model_ridge import predict_game_winner

# --- Load rolling data ---
rolling_df_path = root / "data" / "rolling_df.csv"
rolling_df = pd.read_csv(rolling_df_path)

# Load saved artifacts
models_folder = root / "models"
model = joblib.load(models_folder / "ridge_classifier_final.pkl")
selected_predictors = joblib.load(models_folder / "selected_predictors_ridge.pkl")

# Predict a hypothetical game
result = predict_game_winner(
    rolling_df=rolling_df,
    selected_predictors=selected_predictors,
    model=model,
    home_team="LAL",
    away_team="LAC",
)

print(f"Predicted winner: {result['predicted_winner']}")
print(f"Predicted loser: {result['predicted_loser']}")
