
import pandas as pd

ROLLING_CSV_PATH = "../data/rolling_df.csv"

def compute_rolling_features(df: pd.DataFrame, predictors, window: int = 10):
    """
    Load NBA data, prepare features, compute rolling averages, and
    return a DataFrame ready for modeling.
    
    Parameters
    ----------
    window : int
        Rolling window size (default=10)
    
    Returns
    -------
    pd.DataFrame
        Original data with rolling features and shifted columns added.
    """

    # Keep only numeric predictors + key columns for rolling
    df_rolling = df[list(predictors) + ["won", "team", "season"]]

    # Rolling function
    def find_team_averages(team: pd.DataFrame):
        team_numeric = team.copy()
        
        # Convert boolean columns to float
        bool_cols = team_numeric.select_dtypes(include="bool").columns
        team_numeric[bool_cols] = team_numeric[bool_cols].astype(float)
        
        # Select numeric columns
        numeric_cols = team_numeric.select_dtypes(include="number").columns
        
        # Compute rolling mean
        rolled = team_numeric[numeric_cols].rolling(window).mean()
        return rolled

    # Compute rolling averages grouped by team and season
    df_rolling = df_rolling.groupby(["team", "season"], group_keys=False).apply(find_team_averages)

    # Rename rolling columns to include window suffix
    rolling_cols = [f"{col}_{window}" for col in df_rolling.columns]
    df_rolling.columns = rolling_cols

    # Combine original df with rolling features
    df = pd.concat([df, df_rolling], axis=1)

    # Drop rows with any NaNs created by rolling
    df = df.dropna()

    # Helper functions for shifting columns
    def shift_col(team, col_name):
        return team[col_name].shift(-1)

    def add_col(df, col_name):
        return df.groupby("team", group_keys=False).apply(lambda x: shift_col(x, col_name))

    # Add next-game columns
    df["home_next"] = add_col(df, "home")
    df["team_opp_next"] = add_col(df, "team_opp")
    df["date_next"] = add_col(df, "date")

    # Merge to align rolling features with opponent's next game
    full = df.merge(
        df[rolling_cols + ["team_opp_next", "date_next", "team"]],
        left_on=["team", "date_next"],
        right_on=["team_opp_next", "date_next"],
    )

    rolling_removed_columns = list(full.columns[full.dtypes == "object"])

    full.to_csv(ROLLING_CSV_PATH, index=False)

    print(f"Saved rolling_df as {ROLLING_CSV_PATH}")

    return full, rolling_removed_columns


if __name__ == "__main__":
    from scrape.preprocess_nba_data import load_data
    df = load_data()

    removed_columns = ["season", "date", "won", "target", "team", "team_opp"]
    from ml.train.train_nba_model_ridge import prepare_features
    df, predictors = prepare_features(df, removed_columns)
    df_full, _ = compute_rolling_features(df, predictors)
    print(df_full)
