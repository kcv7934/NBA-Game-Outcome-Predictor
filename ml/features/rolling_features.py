import pandas as pd

ROLLING_CSV_PATH = "../data/rolling_df.csv"

def compute_rolling_features(df: pd.DataFrame, predictors, window: int = 10):
    """
    Compute rolling averages for numeric predictors and prepare a dataset
    suitable for modeling NBA games.

    Steps:
    1. Keep numeric predictors + key columns ('won', 'team', 'season').
    2. Compute rolling averages for each team within each season.
    3. Shift columns to align rolling features with the next game.
    4. Merge team rolling stats with opponent's next game stats.
    5. Save the resulting dataframe to CSV.

    Parameters
    ----------
    df : pandas.DataFrame
        Original NBA game data with all predictors.
    predictors : list
        Columns to include in rolling calculations.
    window : int, optional
        Rolling window size, by default 10.

    Returns
    -------
    full : pandas.DataFrame
        Original dataframe enriched with rolling features and opponent alignment.
    rolling_removed_columns : list
        Columns that are non-numeric / removed from modeling.
    """

    # Keep only numeric predictors + key columns for rolling
    df_rolling = df[list(predictors) + ["won", "team", "season"]]

    # --- Function to compute rolling averages for a single team ---
    def find_team_averages(team: pd.DataFrame):
        team_numeric = team.copy()
        
        # Convert boolean columns to float for rolling computation
        bool_cols = team_numeric.select_dtypes(include="bool").columns
        team_numeric[bool_cols] = team_numeric[bool_cols].astype(float)
        
        # Select numeric columns only
        numeric_cols = team_numeric.select_dtypes(include="number").columns
        
        # Compute rolling mean
        rolled = team_numeric[numeric_cols].rolling(window).mean()
        return rolled

    # Compute rolling averages grouped by team and season
    df_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)

    # Rename rolling columns to include window size
    rolling_cols = [f"{col}_{window}" for col in df_rolling.columns]
    df_rolling.columns = rolling_cols

    # Combine original df with rolling features
    df = pd.concat([df, df_rolling], axis=1)

    # Drop rows with any NaNs created by rolling
    df = df.dropna()

    # --- Helper functions to shift columns for next game alignment ---
    def shift_col(team, col_name):
        return team[col_name].shift(-1)

    def add_col(df, col_name):
        return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

    # Add next-game columns for home, opponent, and date
    df["home_next"] = add_col(df, "home")
    df["team_opp_next"] = add_col(df, "team_opp")
    df["date_next"] = add_col(df, "date")

    # Merge to align rolling features with opponent's next game
    full = df.merge(
        df[rolling_cols + ["team_opp_next", "date_next", "team"]],
        left_on=["team", "date_next"],
        right_on=["team_opp_next", "date_next"],
    )

    # Track non-numeric columns removed for modeling
    rolling_removed_columns = list(full.columns[full.dtypes == "object"])

    # Save rolling dataframe for later use
    full.to_csv(ROLLING_CSV_PATH, index=False)
    print(f"Saved rolling_df to 'data/rolling_df.csv'")

    return full, rolling_removed_columns


if __name__ == "__main__":
    # --- Example usage ---
    from scrape.preprocess_nba_data import load_data
    df = load_data()

    removed_columns = ["season", "date", "won", "target", "team", "team_opp"]

    from ml.train.train_nba_model_ridge import prepare_features

    # Prepare features (scale numeric columns)
    df, predictors = prepare_features(df, removed_columns)

    # Compute rolling features
    df_full, _ = compute_rolling_features(df, predictors)

    print(df_full)
