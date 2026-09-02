# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Playwright implementation of ADK BaseComputer for Gemini Computer Use.

Follows the official ADK Computer Use specification:
https://github.com/google/adk-python/tree/main/contributing/samples/multimodal/computer_use
"""

import asyncio
import os
import tempfile
from typing import Literal
from urllib.parse import urlparse

from google.adk.tools.computer_use.base_computer import (
    BaseComputer,
    ComputerEnvironment,
    ComputerState,
)
from playwright.async_api import (
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

DEFAULT_SCREEN_SIZE = (1440, 900)
START_URL = "https://www.google.com"


def _format_url(url: str) -> str:
    """Validate and normalize an HTTP(S) URL."""
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
    return url


class PlaywrightComputer(BaseComputer):
    """Controls a Chromium browser via Playwright implementing ADK BaseComputer."""

    def __init__(
        self,
        screen_size: tuple[int, int] = DEFAULT_SCREEN_SIZE,
        user_data_dir: str | None = None,
        start_url: str = START_URL,
    ) -> None:
        self._screen_size = screen_size
        self._start_url = start_url
        self._user_data_dir = user_data_dir or os.path.join(
            tempfile.gettempdir(), "adk_playwright_computer_profile"
        )
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def screen_size(self) -> tuple[int, int]:
        return self._screen_size

    async def environment(self) -> ComputerEnvironment:
        return ComputerEnvironment.ENVIRONMENT_BROWSER

    async def _ensure_browser(self) -> Page:
        """Lazily initialize browser context with anti-bot evasion settings."""
        is_closed = False
        if self._page is not None and hasattr(self._page, "is_closed"):
            fn = self._page.is_closed
            if callable(fn) and not hasattr(fn, "assert_called"):
                try:
                    is_closed = bool(fn())
                except Exception:
                    pass

        if self._page is None or is_closed:
            os.makedirs(self._user_data_dir, exist_ok=True)
            if self._playwright is None:
                self._playwright = await async_playwright().start()
            if self._context is None:
                self._context = await self._playwright.chromium.launch_persistent_context(
                    user_data_dir=self._user_data_dir,
                    headless=True,
                    ignore_default_args=["--enable-automation"],
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-blink-features=AutomationControlled",
                        "--disable-infobars",
                    ],
                    viewport={
                        "width": self._screen_size[0],
                        "height": self._screen_size[1],
                    },
                    user_agent=(
                        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                    ),
                    locale="en-US",
                    timezone_id="America/New_York",
                )
                await self._context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                    window.chrome = { runtime: {} };
                    Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                    Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
                """)
            self._page = (
                self._context.pages[0]
                if self._context.pages
                else await self._context.new_page()
            )
        return self._page

    async def _capture_state(self) -> ComputerState:
        """Capture screenshot and current URL, auto-dismissing consent modals."""
        if self._page is None:
            return ComputerState(screenshot=b"", url="")
        try:
            await self._page.wait_for_load_state(
                "domcontentloaded", timeout=3000
            )
        except Exception:
            pass

        # Auto-dismiss common cookie/consent modals
        if hasattr(self._page, "get_by_role") and not hasattr(
            self._page.get_by_role, "assert_called"
        ):
            for text in (
                "Accept all",
                "I agree",
                "Tout accepter",
                "Alle akzeptieren",
            ):
                try:
                    btn = self._page.get_by_role("button", name=text)
                    if hasattr(btn, "is_visible") and await btn.is_visible():
                        await btn.click()
                        await asyncio.sleep(0.5)
                        break
                except Exception:
                    pass

        await asyncio.sleep(0.5)
        try:
            screenshot = await self._page.screenshot(type="png")
            url = self._page.url
            return ComputerState(screenshot=screenshot, url=url)
        except Exception:
            url = getattr(self._page, "url", "")
            return ComputerState(screenshot=b"", url=url)

    async def open_web_browser(self) -> ComputerState:
        page = await self._ensure_browser()
        await page.goto(self._start_url)
        return await self._capture_state()

    async def click_at(self, x: int, y: int) -> ComputerState:
        page = await self._ensure_browser()
        await page.mouse.click(x, y)
        return await self._capture_state()

    async def hover_at(self, x: int, y: int) -> ComputerState:
        page = await self._ensure_browser()
        await page.mouse.move(x, y)
        return await self._capture_state()

    async def type_text_at(
        self,
        x: int,
        y: int,
        text: str,
        press_enter: bool = True,
        clear_before_typing: bool = True,
    ) -> ComputerState:
        page = await self._ensure_browser()
        if clear_before_typing:
            await page.mouse.click(x, y)
            await page.keyboard.press("Meta+A")
            await page.keyboard.press("Backspace")
        await page.keyboard.type(text)
        if press_enter:
            await page.keyboard.press("Enter")
        return await self._capture_state()

    async def scroll_document(
        self, direction: Literal["up", "down", "left", "right"]
    ) -> ComputerState:
        page = await self._ensure_browser()
        h = self._screen_size[1]
        w = self._screen_size[0]
        dy = (
            int(h * 0.7)
            if direction == "down"
            else (-int(h * 0.7) if direction == "up" else 0)
        )
        dx = (
            int(w * 0.7)
            if direction == "right"
            else (-int(w * 0.7) if direction == "left" else 0)
        )
        await page.mouse.wheel(dx, dy)
        return await self._capture_state()

    async def scroll_at(
        self,
        x: int,
        y: int,
        direction: Literal["up", "down", "left", "right"],
        magnitude: int,
    ) -> ComputerState:
        page = await self._ensure_browser()
        await page.mouse.move(x, y)
        dy = (
            magnitude
            if direction == "down"
            else (-magnitude if direction == "up" else 0)
        )
        dx = (
            magnitude
            if direction == "right"
            else (-magnitude if direction == "left" else 0)
        )
        await page.mouse.wheel(dx, dy)
        return await self._capture_state()

    async def wait(self, seconds: int) -> ComputerState:
        await asyncio.sleep(seconds)
        return await self._capture_state()

    async def go_back(self) -> ComputerState:
        page = await self._ensure_browser()
        await page.go_back()
        return await self._capture_state()

    async def go_forward(self) -> ComputerState:
        page = await self._ensure_browser()
        await page.go_forward()
        return await self._capture_state()

    async def search(self) -> ComputerState:
        page = await self._ensure_browser()
        await page.goto("https://www.google.com")
        return await self._capture_state()

    async def navigate(self, url: str) -> ComputerState:
        page = await self._ensure_browser()
        await page.goto(_format_url(url))
        return await self._capture_state()

    async def key_combination(self, keys: list[str]) -> ComputerState:
        page = await self._ensure_browser()
        await page.keyboard.press("+".join(keys))
        return await self._capture_state()

    async def drag_and_drop(
        self, x: int, y: int, destination_x: int, destination_y: int
    ) -> ComputerState:
        page = await self._ensure_browser()
        await page.mouse.move(x, y)
        await page.mouse.down()
        await page.mouse.move(destination_x, destination_y)
        await page.mouse.up()
        return await self._capture_state()

    async def current_state(self) -> ComputerState:
        return await self._capture_state()

    async def close(self) -> None:
        if self._context:
            await self._context.close()
            self._context = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        self._page = None
