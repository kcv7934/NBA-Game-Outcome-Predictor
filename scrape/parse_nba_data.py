import os
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
from tqdm import tqdm

from read_nba_seasons import SCORES_DIR

EXPECTED_COLS = 153

def get_box_scores():
    return [
        os.path.join(SCORES_DIR, f)
        for f in os.listdir(SCORES_DIR)
        if f.endswith(".html")
    ]


def parse_html(box_score):
    with open(box_score, encoding="utf-8", errors="replace") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    [s.decompose() for s in soup.select("tr.over_header")]
    [s.decompose() for s in soup.select("tr.thead")]
    return soup


def read_line_score(soup):
    line_score = pd.read_html(
        StringIO(str(soup)),
        attrs={"id": "line_score"}
    )[0]

    cols = list(line_score.columns)
    cols[0] = "team"
    cols[-1] = "total"
    line_score.columns = cols

    return line_score[["team", "total"]]


def read_stats(soup, team, stat):
    df = pd.read_html(
        StringIO(str(soup)),
        attrs={"id": f"box-{team}-game-{stat}"},
        index_col=0
    )[0]

    return df.apply(pd.to_numeric, errors="coerce")


def read_season_info(soup):
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"] for a in nav.find_all("a")]
    return os.path.basename(hrefs[1]).split("_")[0]


def build_team_summary(soup, team, base_cols):
    basic = read_stats(soup, team, "basic")
    advanced = read_stats(soup, team, "advanced")

    totals = pd.concat([basic.iloc[-1, :], advanced.iloc[-1, :]])
    totals.index = totals.index.str.lower()

    maxes = pd.concat([
        basic.iloc[:-1, :].max(),
        advanced.iloc[:-1, :].max()
    ])
    maxes.index = maxes.index.str.lower() + "_max"

    summary = pd.concat([totals, maxes])

    if base_cols is None:
        base_cols = list(summary.index.drop_duplicates(keep="first"))
        base_cols = [c for c in base_cols if "bpm" not in c]

    return summary[base_cols], base_cols


def build_game(soup, box_score, base_cols):
    line_score = read_line_score(soup)
    teams = list(line_score["team"])

    summaries = []
    for team in teams:
        summary, base_cols = build_team_summary(soup, team, base_cols)
        summaries.append(summary)

    summary_df = pd.concat(summaries, axis=1).T
    game = pd.concat([summary_df, line_score], axis=1)

    game["home"] = [0, 1]

    game_opp = game.iloc[::-1].reset_index(drop=True)
    game_opp.columns += "_opp"

    full_game = pd.concat([game, game_opp], axis=1)

    full_game["season"] = read_season_info(soup)
    full_game["date"] = pd.to_datetime(
        os.path.basename(box_score)[:8],
        format="%Y%m%d"
    )
    full_game["won"] = full_game["total"] > full_game["total_opp"]

    if full_game.shape[1] != EXPECTED_COLS:
        print(f"Skipping {box_score}: bad column count")
        return None, base_cols

    return full_game, base_cols


def main():
    box_scores = get_box_scores()
    games = []
    base_cols = None

    for box_score in tqdm(box_scores, desc="Parsing games"):
        soup = parse_html(box_score)
        game, base_cols = build_game(soup, box_score, base_cols)
        if game is None:
            continue
        games.append(game)

    games_df = pd.concat(games, ignore_index=True)
    games_df.to_csv("nba_games.csv", index=False)

    return games_df


if __name__ == "__main__":
    main()
