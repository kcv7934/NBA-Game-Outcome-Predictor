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
    """Scale the features and return X, y for ML training."""

    # Select predictors
    selected_columns = df.columns[~df.columns.isin(removed_columns)]

    # Scale features
    scaler = MinMaxScaler()
    df[selected_columns] = scaler.fit_transform(df[selected_columns])

    return df, selected_columns


def select_features(df: pd.DataFrame, predictors, target="target", n_features=30, n_splits=3):
    """Use Sequential Feature Selector to pick top features."""
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
    Backtest a model across seasons.

    - df: input DataFrame
    - model: sklearn model
    - predictors: list of feature columns
    - start: starting season index
    - step: season step size
    """
    all_predictions = []
    seasons = sorted(df["season"].unique())

    for i in range(start, len(seasons), step):
        season = seasons[i]

        train = df[df["season"] < season]
        test = df[df["season"] == season]

        model.fit(train[predictors], train["target"])
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)

        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["actual", "prediction"]

        all_predictions.append(combined)

    return pd.concat(all_predictions)



def evaluate_backtest(predictions: pd.DataFrame):
    """Calculate accuracy excluding future/unknown games (target=2)."""
    valid_preds = predictions[predictions["actual"] != 2]
    acc = accuracy_score(valid_preds["actual"], valid_preds["prediction"])
    return acc

def predict_game_winner(rolling_df, selected_predictors, model, home_team, away_team):
    """
    Predict the winner of a single game using RidgeClassifier.

    Returns
    -------
    dict
        Predicted winner and loser
    """

    matchup = rolling_df[
        (rolling_df["team_x"] == home_team) &
        (rolling_df["team_y"] == away_team)
    ]

    if matchup.empty:
        raise ValueError(f"No matchup data for {home_team} vs {away_team}")

    # Use most recent available matchup
    latest_row = matchup.sort_values("date").iloc[-1]

    X = pd.DataFrame(
        [latest_row[selected_predictors].values],
        columns=selected_predictors
    )

    pred = model.predict(X)[0]

    # By convention: target == 1 → team_x wins (home)
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

    # Load cleaned data 
    df = preprocess.load_data()

    removed_columns = ["season", "date", "won", "target", "team", "team_opp"]

    # Prepare and scale features

    # Use improved features via rolling

    df, predictors = prepare_features(df, removed_columns)

    rolling_df, rolling_removed_columns = compute_rolling_features(df, predictors, window=10)

    removed_columns = rolling_removed_columns + removed_columns

    rolling_predictors = rolling_df.columns[~rolling_df.columns.isin(removed_columns)]

    # Feature selection
    selected_predictors = select_features(rolling_df, rolling_predictors)

    # --- Save predictors and model to models folder ---
    models_folder = root / "models"

    joblib.dump(selected_predictors, models_folder / "selected_predictors_ridge.pkl")

    # Backtest model
    model = RidgeClassifier(alpha=1)
    predictions = backtest(rolling_df, model, selected_predictors)

    # Evaluate backtest accuracy
    backtest_acc = evaluate_backtest(predictions)
    print(f"Backtest Accuracy: {backtest_acc:.4f}")

    # Train final model on all data
    final_model = RidgeClassifier(alpha=1)
    final_model.fit(rolling_df[selected_predictors], rolling_df["target"])

    # Evaluate final model on all data
    final_preds = final_model.predict(rolling_df[selected_predictors])
    final_acc = accuracy_score(rolling_df["target"], final_preds)
    print(f"Final Model In-Sample Accuracy: {final_acc:.4f}")

    # Save final model
    joblib.dump(final_model, models_folder / "ridge_classifier_final.pkl")
    print(f"Final model saved as '{models_folder}'")
