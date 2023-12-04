
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

async def navigate_to_url(url):
    """
    Navigate to the specified URL using the provided browser instance.

    Parameters:
    - url: The URL to navigate to.
    """
    global browser
    if browser is None:
        await setup_browser()
    page = await browser.new_page()
    await page.goto(url)
    print(await page.content()) 


list_of_tools = {
    "navigate_to_url": {
        "function": "navigate_to_url",
        "description": "Navigate to the specified URL using the provided browser instance.",
        "parameters": {
            "url": "The URL to navigate to. Must include http:// or https://"
        }
    },
}