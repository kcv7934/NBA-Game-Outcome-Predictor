"""
parse_nba_games.py

This script parses downloaded NBA box score HTML files (from Basketball-Reference)
and converts them into a structured dataset for analysis.

Workflow:
1. Load all saved box score HTML files from SCORES_DIR.
2. Parse each file with BeautifulSoup.
3. Extract team statistics (basic + advanced), totals, and max player values.
4. Combine both teams into a single game-level record.
5. Add metadata such as season, date, home/away flag, and win indicator.
6. Save the final dataset as data/nba_games.csv.

Expected output shape is validated against EXPECTED_COLS. Files that do not
match are skipped.

NOTE: EXPECTED_COLS may vary depending on first season extracted. Recommended to take first
box score of game saved to data/scores and use the column count of that as EXPECTED_COLS
"""

import os
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO
from tqdm import tqdm

from read_nba_seasons import SCORES_DIR

# Expected number of columns per parsed game record
EXPECTED_COLS = 153


def get_box_scores():
    """
    Return a list of full file paths to all saved box score HTML files.

    Returns
    -------
    list of str
        Paths to *.html files inside SCORES_DIR.
    """
    return [
        os.path.join(SCORES_DIR, f)
        for f in os.listdir(SCORES_DIR)
        if f.endswith(".html")
    ]


def parse_html(box_score):
    """
    Load and parse a box score HTML file into a BeautifulSoup object.

    Parameters
    ----------
    box_score : str
        Full path to a box score HTML file.

    Returns
    -------
    BeautifulSoup
        Parsed HTML with header rows removed for easier table parsing.
    """
    with open(box_score, encoding="utf-8", errors="replace") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Remove duplicate header rows that break pandas
    [s.decompose() for s in soup.select("tr.over_header")]
    [s.decompose() for s in soup.select("tr.thead")]

    return soup


def read_line_score(soup):
    """
    Read the game line score table containing team names and totals.

    Parameters
    ----------
    soup : BeautifulSoup

    Returns
    -------
    pandas.DataFrame with columns:
        team  - team name string
        total - total points scored
    """
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
    """
    Read either 'basic' or 'advanced' box score tables for a team.

    Parameters
    ----------
    soup : BeautifulSoup
    team : str
    stat : str
        Either 'basic' or 'advanced'

    Returns
    -------
    pandas.DataFrame
        Numeric player statistics table.
    """
    df = pd.read_html(
        StringIO(str(soup)),
        attrs={"id": f"box-{team}-game-{stat}"},
        index_col=0
    )[0]

    return df.apply(pd.to_numeric, errors="coerce")


def read_season_info(soup):
    """
    Extract the NBA season year from navigation links.

    Parameters
    ----------
    soup : BeautifulSoup

    Returns
    -------
    str
        Season end year (e.g., '2024')
    """
    nav = soup.select("#bottom_nav_container")[0]
    hrefs = [a["href"] for a in nav.find_all("a")]
    return os.path.basename(hrefs[1]).split("_")[0]


def build_team_summary(soup, team, base_cols):
    """
    Build a summary vector of totals and max player stats for one team.

    Parameters
    ----------
    soup : BeautifulSoup
    team : str
    base_cols : list or None
        Column order to enforce across games. Determined on first game.

    Returns
    -------
    (pandas.Series, list)
        Summary stats and final ordered column list.
    """
    basic = read_stats(soup, team, "basic")
    advanced = read_stats(soup, team, "advanced")

    # Team totals = last row
    totals = pd.concat([basic.iloc[-1, :], advanced.iloc[-1, :]])
    totals.index = totals.index.str.lower()

    # Max player stats = max over player rows
    maxes = pd.concat([
        basic.iloc[:-1, :].max(),
        advanced.iloc[:-1, :].max()
    ])
    maxes.index = maxes.index.str.lower() + "_max"

    summary = pd.concat([totals, maxes])

    # Build baseline column list on first run
    if base_cols is None:
        base_cols = list(summary.index.drop_duplicates(keep="first"))
        base_cols = [c for c in base_cols if "bpm" not in c]

    return summary[base_cols], base_cols


def build_game(soup, box_score, base_cols):
    """
    Build a complete game record combining both teams.

    Parameters
    ----------
    soup : BeautifulSoup
    box_score : str
        Source filename (used to infer date)
    base_cols : list or None

    Returns
    -------
    (pandas.DataFrame or None, list)
        Full game record (2 rows, one per team) or None on failure,
        along with the maintained column list.
    """
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
    """
    Main entry point for parsing all saved NBA box scores.

    Returns
    -------
    pandas.DataFrame
        Combined dataset of all parsed games.
    """
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
    games_df.to_csv("data/nba_games.csv", index=False)

    return games_df


if __name__ == "__main__":
    main()
