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
    Combine Ridge (winner) + Logistic (probability) with tiny weighting
    to align probabilities with predicted winner.
    """

    matchup = rolling_df[
        (rolling_df["team_x"] == home_team) &
        (rolling_df["team_y"] == away_team)
    ]

    if matchup.empty:
        raise ValueError(f"No matchup data for {home_team} vs {away_team}")

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

    # --- Logistic probability ---
    X_log = pd.DataFrame(
        [latest_row[logistic_predictors].values],
        columns=logistic_predictors
    )
    probs = logistic_model.predict_proba(X_log)[0]
    home_win_prob = probs[1]

    # Apply tiny weight adjustment
    home_win_prob += ridge_adj
    # Keep within [0,1]
    home_win_prob = min(max(home_win_prob, 0), 1)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_winner": winner,
        "predicted_loser": loser,
        "home_win_prob": home_win_prob,
        "away_win_prob": 1 - home_win_prob,
    }
