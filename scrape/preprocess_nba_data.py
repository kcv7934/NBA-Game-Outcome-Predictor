"""
preprocess_nba_data.py

This module loads the processed NBA games dataset and prepares it
for modeling or further analysis.

Workflow:
1. Load data/nba_games.csv
2. Sort rows chronologically
3. Remove unused/duplicate minute-played columns (only keep 'mp' as all columns share the same value)
4. Create a predictive target column:
      target = whether the team wins its NEXT game
5. Handle free throw % missing values   (There is one game in either 2023 or 2024 season where one team did not shoot any freethrows)
6. Drop any columns that still contain null values
7. Return a clean DataFrame

The resulting dataset is suitable for machine learning workflows.
"""

import pandas as pd

# Path to the parsed NBA games dataset
NBA_GAMES_PATH = "data/nba_games.csv"


def load_data():
    """
    Load, clean, and prepare the NBA games dataset.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataset with target variable created.

    Behavior
    --------
    - Sorts rows by date.
    - Drops unneeded MP (minutes played) columns.
    - Constructs a 'target' label indicating the outcome of each team's next game:
          1 = win next game
          0 = lose next game
          2 = no next game available
    - Replaces NaN FT% stats with zero.
    - Removes any columns that contain remaining null values.
    """

    # Load and sort chronologically
    df = (
        pd.read_csv(NBA_GAMES_PATH)
          .sort_values("date")
          .reset_index(drop=True)
    )

    # Remove unused MP-related columns
    df = df.drop(columns=[
        "mp.1",
        "mp_opp",
        "mp_opp.1",
        "mp_max.1",
        "mp_max_opp.1",
    ])

    # Target = next game's win/loss for same team
    df["target"] = df.groupby("team")["won"].shift(-1)

    # Mark missing as 2 (no next game)
    df.loc[df["target"].isnull(), "target"] = 2
    df["target"] = df["target"].astype(int, errors="ignore")

    # Replace missing FT% with zero
    ft_cols = ["ft%", "ft%_max", "ft%_max_opp"]
    df[ft_cols] = df[ft_cols].fillna(0)

    # Identify and remove any columns still containing NaNs
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0].index
    valid_columns = df.columns[~df.columns.isin(null_cols)]
    df = df[valid_columns].copy()

    return df
