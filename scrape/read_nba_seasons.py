import os
from fetch_nba_seasons import DATA_DIR, STANDINGS_DIR, get_html
import asyncio 
from bs4 import BeautifulSoup

SCORES_DIR = os.path.join(DATA_DIR, "scores")
os.makedirs(SCORES_DIR, exist_ok=True)

standings_files = os.listdir(STANDINGS_DIR)
standings_files = [s for s in standings_files if ".html" in s]

async def scrape_game(standings_file):
    with open(standings_file, 'r', encoding="utf-8", errors="replace") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a")
    hrefs = [l.get("href") for l in links]
    box_scores = [l for l in hrefs if l and "boxscore" in l and ".html" in l]
    box_scores = [f"https://www.basketball-reference.com{L}" for L in box_scores]

    if not box_scores:
        print(f"No boxscores found in {standings_file}")
        return

    for url in box_scores:
        save_path = os.path.join(SCORES_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            print(f"Skipping {save_path}, already exists.")
            continue

        html = await get_html(url, "#content")
        if not html:
            continue

        with open(save_path, "w+", encoding="utf-8", errors="replace") as f:
            f.write(html)

async def scrape_games():
    for f in standings_files:
        filepath = os.path.join(STANDINGS_DIR, f)
        await scrape_game(filepath)

if __name__ == "__main__":
    asyncio.run(scrape_games())