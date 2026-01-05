"""
train_nba_model_log_reg.py

This script trains a logistic regression model to predict NBA game outcomes
using rolling historical team statistics.

Pipeline:
1. Load prepared NBA game data
2. Scale numerical features
3. Compute rolling team vs opponent features
4. Select the best predictive features using Sequential Feature Selection
5. Train a final Logistic Regression classifier on all available data
6. Save the trained model and selected predictors to disk

The model predicts:
    Probability(team wins next game)

Outputs saved in:
    models/selected_predictors_logistic.pkl
    models/logistic_model_final.pkl
"""

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import sys
from pathlib import Path
import joblib

# Setup project root path
root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from features.rolling_features import compute_rolling_features


def prepare_features(df: pd.DataFrame, removed_columns):
    """
    Scale numeric features and return the transformed DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing all team game records.
    removed_columns : list
        Columns to exclude from scaling.

    Returns
    -------
    (pandas.DataFrame, pandas.Index)
        Scaled DataFrame and list of predictor column names.
    """
    selected_columns = df.columns[~df.columns.isin(removed_columns)]
    scaler = MinMaxScaler()
    df[selected_columns] = scaler.fit_transform(df[selected_columns])
    return df, selected_columns


def select_features(df: pd.DataFrame, predictors, target="target",
                    n_features=30, n_splits=3):
    """
    Select the most predictive features using Sequential Forward Selection.

    Parameters
    ----------
    df : pandas.DataFrame
    predictors : list-like
        Candidate predictor columns.
    target : str
        Name of target column.
    n_features : int
        Number of features to select.
    n_splits : int
        Number of time-series CV splits.

    Returns
    -------
    list
        Selected predictor names.
    """
    model = LogisticRegression(max_iter=1000)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=n_features,
        direction="forward",
        cv=tscv
    )

    sfs.fit(df[predictors], df[target])
    selected_predictors = list(predictors[sfs.get_support()])

    return selected_predictors


def train_final_model(df, selected_predictors):
    """
    Train the logistic regression classifier on the full dataset.

    Parameters
    ----------
    df : pandas.DataFrame
    selected_predictors : list

    Returns
    -------
    sklearn.linear_model.LogisticRegression
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(df[selected_predictors], df["target"])
    return model


def predict_game_probabilities(rolling_df, selected_predictors, model,
                               home_team, away_team):
    """
    Predict the probability the home team wins a single matchup.

    Parameters
    ----------
    rolling_df : pandas.DataFrame
        Dataset containing rolling team statistics.
    selected_predictors : list
        Predictor column names.
    model : sklearn model
        Trained classifier with predict_proba().
    home_team : str
    away_team : str

    Returns
    -------
    dict
        {
            "home_team": str,
            "away_team": str,
            "home_win_prob": float,
            "away_win_prob": float
        }
    """

    matchup = rolling_df[
        (rolling_df["team_x"] == home_team) &
        (rolling_df["team_y"] == away_team)
    ]

    if matchup.empty:
        raise ValueError(f"No matchup data for {home_team} vs {away_team}")

    latest_row = matchup.sort_values("date").iloc[-1]

    X = pd.DataFrame(
        [latest_row[selected_predictors].values],
        columns=selected_predictors
    )

    probs = model.predict_proba(X)[0]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_prob": probs[1],
        "away_win_prob": probs[0],
    }


if __name__ == "__main__":
    import scrape.preprocess_nba_data as preprocess

    # Load and scale features
    df = preprocess.load_data()
    removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
    df, predictors = prepare_features(df, removed_columns)

    # Compute rolling features
    rolling_df, rolling_removed_columns = compute_rolling_features(
        df,
        predictors,
        window=10
    )

    rolling_predictors = rolling_df.columns[
        ~rolling_df.columns.isin(removed_columns + rolling_removed_columns)
    ]

    # Feature selection
    selected_predictors = select_features(rolling_df, rolling_predictors)

    # Model storage path
    models_folder = root / "models"

    # Save predictor list
    joblib.dump(selected_predictors, models_folder / "selected_predictors_logistic.pkl")

    # Train final model
    model = train_final_model(rolling_df, selected_predictors)

    # Save trained model
    joblib.dump(model, models_folder / "logistic_model_final.pkl")

    print(f"Saved model as '{models_folder}'")
