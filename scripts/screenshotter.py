import asyncio
import sys
import os
import urllib.parse


async def get_extension_screenshot(extension_name: str, output_path: str) -> bool:
    """Search the Chrome Web Store and screenshot the result page.

    Note: imports playwright lazily because this utility is not part of the main runtime path.
    """
    try:
        from playwright.async_api import async_playwright
    except Exception as e:
        print(
            "Playwright is not installed or not set up. Install with:\n"
            "  pip install playwright\n  playwright install chromium\n"
            f"Details: {e}"
        )
        return False

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        search_query = f"{extension_name} chrome extension"
        search_url = f"https://www.google.com/search?q={urllib.parse.quote(search_query)}"
        await page.goto(search_url)

        links = await page.query_selector_all("a")
        store_url = None
        for link in links:
            href = await link.get_attribute("href")
            if href and (
                "chrome.google.com/webstore/detail" in href
                or "chromewebstore.google.com/detail" in href
            ):
                store_url = href
                break

        if not store_url:
            print(f"Could not find Chrome Web Store link for {extension_name}")
            await browser.close()
            return False

        await page.goto(store_url, wait_until="networkidle")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        await page.screenshot(path=output_path, full_page=False)
        await browser.close()
        return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 screenshotter.py <extension_name> <output_path>")
        sys.exit(1)

    name = sys.argv[1]
    path = sys.argv[2]

    asyncio.run(get_extension_screenshot(name, path))
