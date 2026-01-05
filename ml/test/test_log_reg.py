import pandas as pd
import joblib
from pathlib import Path 
import sys

"""
test_log_reg.py

This script demonstrates how to use the trained Logistic Regression model
to predict the probability that a home team will win an NBA game.

Pipeline:
1. Load rolling game features from CSV.
2. Load the trained logistic regression model and corresponding predictor list.
3. Predict win probabilities for a single hypothetical game (home vs away).
4. Print the predicted probabilities for home and away teams.

Outputs:
- Dictionary containing: 
    'home_team': str
    'away_team': str
    'home_win_prob': float
    'away_win_prob': float
"""

# --- Set project root (one level above test/) ---
# Resolve the absolute path to the parent directory of this file
# Then add that directory to sys.path so imports work correctly
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

# Import the function used to generate win probability predictions
from train.train_nba_model_log_reg import predict_game_probabilities 

# --- Load rolling data ---
# Load the rolling statistics dataframe used as model input
rolling_df_path = root / "data" / "rolling_df.csv"
rolling_df = pd.read_csv(rolling_df_path)

# --- Load saved model + predictor list ---
# Load the trained logistic regression model from disk
models_folder = root / "models"
model = joblib.load(models_folder / "logistic_model_final.pkl")

# Load the list of predictor columns that the model was trained on
selected_predictors = joblib.load(models_folder / "selected_predictors_logistic.pkl")

# --- Predict a hypothetical game ---
# Call the prediction function to compute win probabilities
# Inputs:
#   rolling_df — rolling team stats
#   selected_predictors — list of features used by the trained model
#   model — trained logistic regression object
#   home_team — home team abbreviation
#   away_team — away team abbreviation
#
# Output:
#   Dictionary with:
#       'home_team'
#       'away_team'
#       'home_win_prob'
#       'away_win_prob'
result = predict_game_probabilities(
    rolling_df=rolling_df,
    selected_predictors=selected_predictors,
    model=model,
    home_team="LAL",
    away_team="DET",
)

# --- Display prediction results ---
# Print formatted win probability values for both teams
print(f"{result['home_team']} win probability: {result['home_win_prob']:.2%}")
print(f"{result['away_team']} win probability: {result['away_win_prob']:.2%}")
