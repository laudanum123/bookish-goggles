

from playwright.async_api import async_playwright


browser = None


async def setup_browser():
    """

    Set up and return a Playwright browser instance.
    """
    global browser

    if browser is None:

        playwright = await async_playwright().start()

        browser = await playwright.chromium.launch(headless=False)


async def suche_artikel(item: str):
    """
    Search for an item in a webstore.

    Parameters:
    - item: The item to search for.
    """
    global browser

    if browser is None:
        await setup_browser()

    page = await browser.new_page()
    await page.goto(f"https://momo.abo-kiste.com/web/main.php/shop/suche?suchtext={item}")

    # Search for elements by CSS selector
    elements = await page.query_selector_all(".produktListe_text")

    inner_html_list = []
    for element in elements:
        inner_html = await element.inner_html()
        inner_html_list.append(inner_html)
        print(inner_html)

    return elements



list_of_tools = {

    "suche_artikel": {
        "function": "suche_artikel",
        "beschreibung": "Suche nach einem Artikel in einem Webshop.",
        "artikel": "artikel_name"

    },

}