import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from features.rolling_features import compute_rolling_features
import joblib

def prepare_features(df: pd.DataFrame, removed_columns):
    """Scale the features and return X, y for ML training."""
    selected_columns = df.columns[~df.columns.isin(removed_columns)]
    scaler = MinMaxScaler()
    df[selected_columns] = scaler.fit_transform(df[selected_columns])
    return df, selected_columns

def select_features(df: pd.DataFrame, predictors, target="target", n_features=30, n_splits=3):
    """Use Sequential Feature Selector to pick top features."""
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
    """Train final classifier on all data."""
    model = LogisticRegression(max_iter=1000)
    model.fit(df[selected_predictors], df["target"])
    return model

def predict_game_probabilities(rolling_df, selected_predictors, model, home_team, away_team):
    """
    Predict the probability that the home team wins a single game
    using the most recent available rolling features.

    Parameters
    ----------
    rolling_df : pd.DataFrame
        Data with rolling features (already merged team vs opponent)
    selected_predictors : list
        Columns used for prediction
    model : sklearn classifier
        Trained model with predict_proba
    home_team : str
    away_team : str

    Returns
    -------
    dict
        Home and away win probabilities
    """

    # Find most recent row for this matchup
    matchup = rolling_df[
        (rolling_df["team_x"] == home_team) &
        (rolling_df["team_y"] == away_team)
    ]

    if matchup.empty:
        raise ValueError(f"No matchup data for {home_team} vs {away_team}")

    latest_row = matchup.sort_values("date").iloc[-1]

    # Build feature matrix
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

    # Load data and prepare features
    df = preprocess.load_data()
    removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
    df, predictors = prepare_features(df, removed_columns)

    rolling_df, rolling_removed_columns = compute_rolling_features(df, predictors, window=10)
    rolling_predictors = rolling_df.columns[~rolling_df.columns.isin(removed_columns + rolling_removed_columns)]

    # Feature selection
    selected_predictors = select_features(rolling_df, rolling_predictors)

    joblib.dump(selected_predictors, "selected_predictors_logistic.pkl")


    # Train final model
    model = train_final_model(rolling_df, selected_predictors)

    # Save model
    joblib.dump(model, "logistic_model_final.pkl")
    print("Saved model as 'logistic_model_final.pkl'")

