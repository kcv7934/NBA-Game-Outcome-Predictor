import pandas as pd
import joblib
from predictor.ensemble_predictor import predict_game_ensemble_weighted
from pathlib import Path
import argparse

"""
predict_game.py

This script predicts a single NBA game's outcome using the ensemble predictor
that combines RidgeClassifier (winner) and LogisticRegression (probability).

Pipeline:
1. Load rolling game features from CSV.
2. Load trained Ridge and Logistic models with predictor lists.
3. Predict a single NBA game specified via command-line arguments.
4. Print predicted winner, loser, and probabilities.
"""

# --- Parse command-line arguments ---
parser = argparse.ArgumentParser(description="Predict NBA game outcome using ensemble model")
parser.add_argument("home_team", type=str, help="Abbreviation of the home team (e.g., LAL)")
parser.add_argument("away_team", type=str, help="Abbreviation of the away team (e.g., BOS)")
args = parser.parse_args()

home_team = args.home_team.upper()
away_team = args.away_team.upper()

# --- Set paths ---
root = Path(__file__).resolve().parent
data_path = root / "data" / "rolling_df.csv"

# --- Load rolling data ---
rolling_df = pd.read_csv(data_path)

# --- Load trained models and predictor columns ---
models_folder = root / "models"

ridge_model = joblib.load(models_folder / "ridge_classifier_final.pkl")
ridge_predictors = joblib.load(models_folder / "selected_predictors_ridge.pkl")

logistic_model = joblib.load(models_folder / "logistic_model_final.pkl")
logistic_predictors = joblib.load(models_folder / "selected_predictors_logistic.pkl")

# --- Predict the game ---
result = predict_game_ensemble_weighted(
    rolling_df=rolling_df,
    ridge_model=ridge_model,
    ridge_predictors=ridge_predictors,
    logistic_model=logistic_model,
    logistic_predictors=logistic_predictors,
    home_team=home_team,
    away_team=away_team,
    ridge_weight=0.02
)

# --- Display results ---
print(f"Home team: {result['home_team']}")
print(f"Away team: {result['away_team']}")
print(f"Winner: {result['predicted_winner']}")
print(f"Loser: {result['predicted_loser']}")
print(f"Home win probability: {result['home_win_prob']:.2%}")
print(f"Away win probability: {result['away_win_prob']:.2%}")
