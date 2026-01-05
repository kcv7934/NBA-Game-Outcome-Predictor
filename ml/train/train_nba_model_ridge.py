"""
train_nba_model_ridge.py

This script trains a Ridge Classifier model to predict NBA game outcomes
using rolling historical team statistics.

Pipeline:
1. Load prepared NBA game data
2. Scale numerical features
3. Compute rolling team vs opponent features
4. Select the best predictive features using Sequential Feature Selection
5. Backtest accuracy across seasons
6. Train a final Ridge Classifier on all available data
7. Save the trained model and selected predictors to disk

The model predicts:
    Winner(team_x vs team_y)

Outputs saved in:
    models/selected_predictors_ridge.pkl
    models/ridge_classifier_final.pkl
"""

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from features.rolling_features import compute_rolling_features
import joblib


def prepare_features(df: pd.DataFrame, removed_columns):
    """
    Scale numeric feature columns and return the transformed DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataset containing team game records.
    removed_columns : list
        Columns to exclude from scaling and modeling.

    Returns
    -------
    (pandas.DataFrame, pandas.Index)
        Scaled dataframe and list of predictor column names.
    """

    # Select predictors only
    selected_columns = df.columns[~df.columns.isin(removed_columns)]

    # Scale predictors into 0–1 range
    scaler = MinMaxScaler()
    df[selected_columns] = scaler.fit_transform(df[selected_columns])

    return df, selected_columns


def select_features(df: pd.DataFrame, predictors,
                    target="target", n_features=30, n_splits=3):
    """
    Select the most predictive features using Sequential Forward Selection.

    Parameters
    ----------
    df : pandas.DataFrame
        Training dataset.
    predictors : list-like
        Candidate predictor columns.
    target : str
        Name of target label column.
    n_features : int
        Number of features to select.
    n_splits : int
        Number of time-series CV folds.

    Returns
    -------
    list
        Selected predictor names.
    """

    model = RidgeClassifier(alpha=1)
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


def backtest(df: pd.DataFrame, model, predictors, start=2, step=1):
    """
    Perform walk-forward backtesting across NBA seasons.

    Trains on all seasons before season N,
    then evaluates on season N, repeating forward.

    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset across seasons.
    model : sklearn estimator
        Ridge classifier instance.
    predictors : list
        Feature column names.
    start : int
        Starting season index to test.
    step : int
        Number of seasons to skip per iteration.

    Returns
    -------
    pandas.DataFrame
        Combined predictions with columns:
        ['actual', 'prediction']
    """

    all_predictions = []
    seasons = sorted(df["season"].unique())

    for i in range(start, len(seasons), step):
        season = seasons[i]

        # Train on all prior seasons
        train = df[df["season"] < season]

        # Test on current season
        test = df[df["season"] == season]

        model.fit(train[predictors], train["target"])
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)

        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]

        all_predictions.append(combined)

    return pd.concat(all_predictions)


def evaluate_backtest(predictions: pd.DataFrame):
    """
    Compute classification accuracy excluding games
    where the target value is unknown (target = 2).

    Parameters
    ----------
    predictions : pandas.DataFrame

    Returns
    -------
    float
        Accuracy score on valid games.
    """

    valid_preds = predictions[predictions["actual"] != 2]
    acc = accuracy_score(valid_preds["actual"], valid_preds["prediction"])
    return acc


def predict_game_winner(rolling_df, selected_predictors, model,
                        home_team, away_team):
    """
    Predict the winner of a single NBA game matchup.

    Parameters
    ----------
    rolling_df : pandas.DataFrame
        Dataset containing rolling team statistics.
    selected_predictors : list
        Predictor column names.
    model : sklearn estimator
        Trained RidgeClassifier.
    home_team : str
    away_team : str

    Returns
    -------
    dict
        {
          "home_team": str,
          "away_team": str,
          "predicted_winner": str,
          "predicted_loser": str
        }
    """

    matchup = rolling_df[
        (rolling_df["team_x"] == home_team) &
        (rolling_df["team_y"] == away_team)
    ]

    if matchup.empty:
        raise ValueError(f"No matchup data for {home_team} vs {away_team}")

    # Most recent matchup record
    latest_row = matchup.sort_values("date").iloc[-1]

    # Build model input
    X = pd.DataFrame(
        [latest_row[selected_predictors].values],
        columns=selected_predictors
    )

    pred = model.predict(X)[0]

    # 1 = home wins
    if pred == 1:
        winner = home_team
        loser = away_team
    else:
        winner = away_team
        loser = home_team

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_winner": winner,
        "predicted_loser": loser,
    }


if __name__ == "__main__":
    import scrape.preprocess_nba_data as preprocess

    # Load cleaned dataset
    df = preprocess.load_data()

    removed_columns = ["season", "date", "won", "target", "team", "team_opp"]

    # Prepare and scale predictors
    df, predictors = prepare_features(df, removed_columns)

    # Compute rolling statistical features
    rolling_df, rolling_removed_columns = compute_rolling_features(
        df,
        predictors,
        window=10
    )

    removed_columns = rolling_removed_columns + removed_columns

    rolling_predictors = rolling_df.columns[
        ~rolling_df.columns.isin(removed_columns)
    ]

    # Select best predictors
    selected_predictors = select_features(rolling_df, rolling_predictors)

    # Save selected predictors
    models_folder = root / "models"
    joblib.dump(selected_predictors, models_folder / "selected_predictors_ridge.pkl")

    # Backtest model performance
    model = RidgeClassifier(alpha=1)
    predictions = backtest(rolling_df, model, selected_predictors)

    backtest_acc = evaluate_backtest(predictions)
    print(f"Backtest Accuracy: {backtest_acc:.4f}")

    # Train final model on full dataset
    final_model = RidgeClassifier(alpha=1)
    final_model.fit(rolling_df[selected_predictors], rolling_df["target"])

    # Evaluate in-sample performance
    final_preds = final_model.predict(rolling_df[selected_predictors])
    final_acc = accuracy_score(rolling_df["target"], final_preds)
    print(f"Final Model In-Sample Accuracy: {final_acc:.4f}")

    # Save trained model
    joblib.dump(final_model, models_folder / "ridge_classifier_final.pkl")
    print(f"Final model saved as '{models_folder}'")
