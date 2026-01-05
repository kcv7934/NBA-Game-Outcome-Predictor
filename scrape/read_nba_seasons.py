"""
read_nba_games.py

This script scrapes NBA box score pages from Basketball-Reference based on
previously-downloaded standings HTML files.

For each standings file found in STANDINGS_DIR, the script:
1. Parses the HTML to find links containing "boxscore".
2. Builds the full Basketball-Reference URL for each box score.
3. Downloads the HTML for each box score page (if not already saved).
4. Saves the box score HTML files into DATA_DIR/scores.
"""

import os
from fetch_nba_seasons import DATA_DIR, STANDINGS_DIR, get_html
import asyncio
from bs4 import BeautifulSoup

# Directory to store downloaded box score HTML files
SCORES_DIR = os.path.join(DATA_DIR, "scores")
os.makedirs(SCORES_DIR, exist_ok=True)

# Collect all standings HTML files
standings_files = os.listdir(STANDINGS_DIR)
standings_files = [s for s in standings_files if ".html" in s]


async def scrape_game(standings_file):
    """
    Parse a single standings HTML file and download all associated
    Basketball-Reference box score pages.

    Parameters
    ----------
    standings_file : str
        Path to a standings HTML file previously downloaded.

    Behavior
    --------
    - Extracts all <a> tag href links.
    - Filters to only those linking to 'boxscore' pages.
    - Downloads each box score page if not already stored.
    - Saves results into SCORES_DIR.
    """
    with open(standings_file, 'r', encoding="utf-8", errors="replace") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    # Extract all href links
    links = soup.find_all("a")
    hrefs = [l.get("href") for l in links]

    # Filter for box score URLs
    box_scores = [l for l in hrefs if l and "boxscore" in l and ".html" in l]
    box_scores = [f"https://www.basketball-reference.com{L}" for L in box_scores]

    if not box_scores:
        print(f"No boxscores found in {standings_file}")
        return

    # Download and save each box score page
    for url in box_scores:
        save_path = os.path.join(SCORES_DIR, url.split("/")[-1])

        # Skip if already downloaded
        if os.path.exists(save_path):
            print(f"Skipping {save_path}, already exists.")
            continue

        html = await get_html(url, "#content")
        if not html:
            return

        with open(save_path, "w+", encoding="utf-8", errors="replace") as f:
            f.write(html)


async def scrape_games():
    """
    Iterate through all standings HTML files and scrape their
    corresponding box score pages.

    This serves as the main async workflow controller.
    """
    for f in standings_files:
        filepath = os.path.join(STANDINGS_DIR, f)
        await scrape_game(filepath)


if __name__ == "__main__":
    """
    Entry point when running the script directly.

    Executes the asynchronous scraping pipeline.
    """
    asyncio.run(scrape_games())
