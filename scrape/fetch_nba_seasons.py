import os, asyncio, time
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

SEASONS = list(range(2020, 2027))
DATA_DIR = "data"
STANDINGS_DIR = os.path.join(DATA_DIR, "standings")
os.makedirs(STANDINGS_DIR, exist_ok=True)

async def get_html(url, selector, sleep=5, retries=3):
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
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = await get_html(url, "#content .filter")

    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a")
    standings_pages = [f"https://www.basketball-reference.com{l['href']}" for l in links]
    for url in standings_pages:
        save_path = os.path.join(STANDINGS_DIR, url.split("/")[-1])
        if os.path.exists(save_path):
            print(f"Skipping {save_path}, already exists.")
            continue
        html = await get_html(url, "#all_schedule")
        with open(save_path, "w+") as f:
            f.write(html)

async def scrape_seasons():
    for season in SEASONS:
        await(scrape_season(season))

if __name__ == "__main__":
    asyncio.run(scrape_seasons())
