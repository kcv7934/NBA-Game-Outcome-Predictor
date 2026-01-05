import pandas as pd

NBA_GAMES_PATH = "nba_games.csv"

def load_data():

    df = (
        pd.read_csv(NBA_GAMES_PATH)
          .sort_values("date")
          .reset_index(drop=True)
    )

    df = df.drop(columns=[
        "mp.1",
        "mp_opp",
        "mp_opp.1",
        "mp_max.1",
        "mp_max_opp.1",
    ])


    df["target"] = df.groupby("team")["won"].shift(-1)
    df.loc[df["target"].isnull(), "target"] = 2
    df["target"] = df["target"].astype(int, errors="ignore")

    ft_cols = ["ft%", "ft%_max", "ft%_max_opp"]
    df[ft_cols] = df[ft_cols].fillna(0)

    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0].index

    valid_columns = df.columns[~df.columns.isin(null_cols)]
    df = df[valid_columns].copy()

    return df