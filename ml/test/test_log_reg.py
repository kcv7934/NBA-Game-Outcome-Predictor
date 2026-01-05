# predict_game.py
import pandas as pd
import joblib
from pathlib import Path 
import sys

# --- Set project root (one level above test/) ---
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from train.train_nba_model_log_reg import predict_game_probabilities 

# --- Load rolling data ---
rolling_df_path = root / "data" / "rolling_df.csv"
rolling_df = pd.read_csv(rolling_df_path)

# Load saved artifacts
models_folder = root / "models"
model = joblib.load(models_folder / "logistic_model_final.pkl")
selected_predictors = joblib.load(models_folder / "selected_predictors_logistic.pkl")

# Predict a hypothetical game
result = predict_game_probabilities(
    rolling_df=rolling_df,
    selected_predictors=selected_predictors,
    model=model,
    home_team="LAL",
    away_team="DET",
)

print(f"{result['home_team']} win probability: {result['home_win_prob']:.2%}")
print(f"{result['away_team']} win probability: {result['away_win_prob']:.2%}")
