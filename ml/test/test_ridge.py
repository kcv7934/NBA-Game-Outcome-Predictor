import pandas as pd
import joblib
import sys
from pathlib import Path

"""
test_ridge.py

This script demonstrates how to use the trained RidgeClassifier model
to predict the winner of an NBA game.

Pipeline:
1. Load rolling game features from CSV.
2. Load the trained RidgeClassifier model and corresponding predictor list.
3. Predict the winner for a single hypothetical game (home vs away).
4. Print the predicted winner and loser.

Outputs:
- Dictionary containing:
    'home_team': str
    'away_team': str
    'predicted_winner': str
    'predicted_loser': str
"""

# --- Set project root (one level above test/) ---
# Resolve the absolute path of this file, move up one directory,
# and add that directory to sys.path so project imports work
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

# Import the function used to predict a game winner
from train.train_nba_model_ridge import predict_game_winner

# --- Load rolling data ---
# Read the rolling statistics dataframe that is used as model input
rolling_df_path = root / "data" / "rolling_df.csv"
rolling_df = pd.read_csv(rolling_df_path)

# --- Load saved artifacts ---
# Load the trained Ridge Classifier model from disk
models_folder = root / "models"
model = joblib.load(models_folder / "ridge_classifier_final.pkl")

# Load the list of predictor feature names used during model training
selected_predictors = joblib.load(models_folder / "selected_predictors_ridge.pkl")

# --- Predict a hypothetical game ---
# Call the prediction function to determine the expected winner
# Inputs:
#   rolling_df — dataframe containing rolling team stats
#   selected_predictors — feature list used by the trained model
#   model — trained ridge classifier model
#   home_team — home team abbreviation
#   away_team — away team abbreviation
result = predict_game_winner(
    rolling_df=rolling_df,
    selected_predictors=selected_predictors,
    model=model,
    home_team="LAL",
    away_team="LAC",
)

# --- Display prediction results ---
# Print the predicted winner and loser to the console
print(f"Predicted winner: {result['predicted_winner']}")
print(f"Predicted loser: {result['predicted_loser']}")
