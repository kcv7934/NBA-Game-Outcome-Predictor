"""
fetch_nba_seasons.py

This script downloads NBA season schedule/standings pages from
Basketball-Reference for a range of seasons.

Workflow:
1. For each season in SEASONS, load the season's main games page.
2. Parse the page to extract links to each month's schedule page.
3. Download each monthly schedule page individually.
4. Save the HTML files locally inside DATA_DIR/standings.
"""

import os, asyncio, time
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

# Seasons to scrape (NBA season end year format)
SEASONS = list(range(2020, 2027))

# Base data directory structure
DATA_DIR = "data"
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")
os.makedirs(STANDINGS_DIR, exist_ok=True)


async def get_html(url, selector, sleep=5, retries=3):
    """
    Load a webpage using Playwright and extract inner HTML from
    the element matching the given CSS selector.

    Parameters
    ----------
    url : str
        Target webpage URL to fetch.
    selector : str
        CSS selector identifying the main content container.
    sleep : int, optional
        Base delay before each retry (in seconds). Keep at least to 5 to limit 429 errors
    retries : int, optional
        Number of retry attempts on failure.

    Returns
    -------
    str or None
        Extracted HTML content from the selected element,
        or None if retrieval fails after all retries.

    Behavior
    --------
    - Delays before each retry to reduce request rate.
    - Uses a headless Chromium browser.
    - Prints the page title for logging.
    - Retries on PlaywrightTimeout.
    """
    html = None
    for i in range(1, retries + 1):
        time.sleep(sleep * i)
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url)
                print(await page.title())
                html = await page.inner_html(selector)
        except PlaywrightTimeout:
            print(f"Timeout error on {url}, retry {i}")
            continue
        else:
            break
    return html


async def scrape_season(season):
    """
    Scrape all monthly schedule pages for a single NBA season.

    Parameters
    ----------
    season : int
        The NBA season end year (e.g., 2024 for 2023–24 season).

    Behavior
    --------
    - Downloads the main season games page.
    - Extracts all links to monthly standings/schedule pages.
    - Saves each month's page HTML to STANDINGS_DIR.
    - Skips files already downloaded.
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = await get_html(url, "#content .filter")

    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a")

    # Build full URLs for each month schedule page
    standings_pages = [f"https://www.basketball-reference.com{l['href']}" for l in links]

    for url in standings_pages:
        save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1])

        # Skip if already downloaded
        if os.path.exists(save_path):
            print(f"Skipping {save_path}, already exists.")
            continue

        html = await get_html(url, "#all_schedule")

        with open(save_path, "w+") as f:
            f.write(html)


async def scrape_seasons():
    """
    Iterate through all SEASONS and scrape each one sequentially.

    Serves as the main async controller for the scraping process.
    """
    for season in SEASONS:
        await(scrape_season(season))


if __name__ == "__main__":
    """
    Entry point when running the script directly.

    Launches the asynchronous scraping pipeline.
    """
    asyncio.run(scrape_seasons())
