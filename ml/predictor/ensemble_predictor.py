import pandas as pd

def predict_game_ensemble_weighted(
    rolling_df,
    ridge_model,
    ridge_predictors,
    logistic_model,
    logistic_predictors,
    home_team,
    away_team,
    ridge_weight=0.02
):
    """
    Predict an NBA game outcome using a weighted ensemble of Ridge and Logistic models.

    Combines:
        - RidgeClassifier: predicts the winner (team_x vs team_y)
        - LogisticRegression: predicts home win probability

    The Ridge prediction slightly adjusts the Logistic probability
    to align the probability with the predicted winner.

    Parameters
    ----------
    rolling_df : pandas.DataFrame
        Data containing rolling team statistics.
    ridge_model : sklearn estimator
        Trained RidgeClassifier model.
    ridge_predictors : list
        Predictor columns used by the Ridge model.
    logistic_model : sklearn estimator
        Trained LogisticRegression model.
    logistic_predictors : list
        Predictor columns used by the Logistic model.
    home_team : str
        Abbreviation of the home team.
    away_team : str
        Abbreviation of the away team.
    ridge_weight : float, optional
        Small adjustment applied to Logistic probability in the direction
        of the Ridge predicted winner (default is 0.02).

    Returns
    -------
    dict
        {
            "home_team": str,
            "away_team": str,
            "predicted_winner": str,
            "predicted_loser": str,
            "home_win_prob": float,
            "away_win_prob": float
        }
    """

    # --- Filter matchup for the specified teams ---
    matchup = rolling_df[
        (rolling_df["team_x"] == home_team) &
        (rolling_df["team_y"] == away_team)
    ]

    if matchup.empty:
        raise ValueError(f"No matchup data for {home_team} vs {away_team}")

    # Use the most recent available matchup
    latest_row = matchup.sort_values("date").iloc[-1]

    # --- Ridge prediction (winner) ---
    X_ridge = pd.DataFrame(
        [latest_row[ridge_predictors].values],
        columns=ridge_predictors
    )
    ridge_pred = ridge_model.predict(X_ridge)[0]

    if ridge_pred == 1:
        winner = home_team
        loser = away_team
        ridge_adj = ridge_weight
    else:
        winner = away_team
        loser = home_team
        ridge_adj = -ridge_weight

    # --- Logistic probability prediction ---
    X_log = pd.DataFrame(
        [latest_row[logistic_predictors].values],
        columns=logistic_predictors
    )
    probs = logistic_model.predict_proba(X_log)[0]
    home_win_prob = probs[1]

    # Apply tiny weight adjustment based on Ridge prediction
    home_win_prob += ridge_adj

    # Ensure probability stays within [0, 1]
    home_win_prob = min(max(home_win_prob, 0), 1)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_winner": winner,
        "predicted_loser": loser,
        "home_win_prob": home_win_prob,
        "away_win_prob": 1 - home_win_prob,
    }

def predict_row_ensemble(
    row,
    ridge_model,
    ridge_predictors,
    logistic_model,
    logistic_predictors,
    ridge_weight=0.02
):
    X_ridge = pd.DataFrame([row[ridge_predictors].values], columns=ridge_predictors)
    ridge_pred = ridge_model.predict(X_ridge)[0]

    X_log = pd.DataFrame([row[logistic_predictors].values], columns=logistic_predictors)
    home_prob = logistic_model.predict_proba(X_log)[0][1]

    if ridge_pred == 1:
        home_prob += ridge_weight
    else:
        home_prob -= ridge_weight

    home_prob = min(max(home_prob, 0), 1)

    return 1 if home_prob >= 0.5 else 0


def backtest_ensemble(
    rolling_df,
    ridge_predictors,
    logistic_predictors,
    ridge_weight=0.02
):
    from sklearn.linear_model import RidgeClassifier, LogisticRegression
    from sklearn.metrics import accuracy_score

    all_actual = []
    all_preds = []

    seasons = sorted(rolling_df["season"].unique())

    for season in seasons[2:]:
        train = rolling_df[rolling_df["season"] < season]
        test = rolling_df[rolling_df["season"] == season]

        ridge_model = RidgeClassifier(alpha=1)
        ridge_model.fit(train[ridge_predictors], train["target"])

        logistic_model = LogisticRegression(max_iter=1000)
        logistic_model.fit(train[logistic_predictors], train["target"])

        for _, row in test.iterrows():
            if row["target"] == 2:
                continue

            pred = predict_row_ensemble(
                row,
                ridge_model,
                ridge_predictors,
                logistic_model,
                logistic_predictors,
                ridge_weight=ridge_weight
            )

            all_preds.append(pred)
            all_actual.append(row["target"])

    acc = accuracy_score(all_actual, all_preds)
    return acc