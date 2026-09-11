# Copyright 2026 Google LLC
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

"""Playwright implementation of ADK BaseComputer for Gemini Computer Use."""

import asyncio
from typing import Literal
from urllib.parse import quote_plus, urlparse

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


MAX_WAIT_SECONDS = 30


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
        start_url: str = START_URL,
    ) -> None:
        self._screen_size = screen_size
        self._start_url = _format_url(start_url)
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def screen_size(self) -> tuple[int, int]:
        return self._screen_size

    async def environment(self) -> ComputerEnvironment:
        return ComputerEnvironment.ENVIRONMENT_BROWSER

    async def initialize(self) -> None:
        """Initialize the browser, context, and default page."""
        if self._page is not None:
            return
        self._playwright = await async_playwright().start()
        browser = await self._playwright.chromium.launch(headless=True)
        self._context = await browser.new_context(
            viewport={
                "width": self._screen_size[0],
                "height": self._screen_size[1],
            },
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        )
        self._page = await self._context.new_page()
        await self._page.goto(_format_url(self._start_url))

    async def _get_page(self) -> Page:
        if self._page is None:
            await self.initialize()
        return self._page

    async def _capture_state(self) -> ComputerState:
        page = await self._get_page()
        try:
            await page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass
        await asyncio.sleep(0.5)
        try:
            screenshot = await page.screenshot(type="png")
        except Exception:
            screenshot = b""

        raw_url = getattr(page, "url", "") or ""
        if (
            not raw_url
            or raw_url == "about:blank"
            or not raw_url.startswith(("http://", "https://"))
        ):
            raw_url = self._start_url

        return ComputerState(screenshot=screenshot, url=raw_url)

    async def open_web_browser(self) -> ComputerState:
        page = await self._get_page()
        await page.goto(_format_url(self._start_url))
        return await self._capture_state()

    async def click_at(
        self,
        x: int = 0,
        y: int = 0,
        coordinate: list[int] | tuple[int, int] | None = None,
        point: list[int] | tuple[int, int] | None = None,
        button: str = "left",
    ) -> ComputerState:
        page = await self._get_page()
        coord = coordinate or point
        if coord and len(coord) >= 2:
            x, y = coord[0], coord[1]
        await page.mouse.click(int(x), int(y))
        return await self._capture_state()

    async def hover_at(
        self,
        x: int = 0,
        y: int = 0,
        coordinate: list[int] | tuple[int, int] | None = None,
        point: list[int] | tuple[int, int] | None = None,
    ) -> ComputerState:
        page = await self._get_page()
        coord = coordinate or point
        if coord and len(coord) >= 2:
            x, y = coord[0], coord[1]
        await page.mouse.move(int(x), int(y))
        return await self._capture_state()

    async def type_text_at(
        self,
        x: int = 0,
        y: int = 0,
        text: str = "",
        value: str = "",
        query: str = "",
        press_enter: bool = True,
        clear_before_typing: bool = True,
        coordinate: list[int] | tuple[int, int] | None = None,
        point: list[int] | tuple[int, int] | None = None,
    ) -> ComputerState:
        page = await self._get_page()
        coord = coordinate or point
        if coord and len(coord) >= 2:
            x, y = coord[0], coord[1]
        if x or y:
            await page.mouse.click(int(x), int(y))
        if clear_before_typing:
            await page.keyboard.press("Meta+A")
            await page.keyboard.press("Backspace")
        text_to_type = text or value or query
        if text_to_type:
            await page.keyboard.type(str(text_to_type))
        if press_enter:
            await page.keyboard.press("Enter")
        return await self._capture_state()

    async def scroll_document(
        self,
        direction: Literal["up", "down", "left", "right"] = "down",
    ) -> ComputerState:
        page = await self._get_page()
        dy = int(self._screen_size[1] * 0.7) * (
            1 if direction == "down" else -1
        )
        await page.mouse.wheel(0, dy)
        return await self._capture_state()

    async def scroll_at(
        self,
        x: int = 0,
        y: int = 0,
        direction: Literal["up", "down", "left", "right"] = "down",
        magnitude: int = 300,
    ) -> ComputerState:
        page = await self._get_page()
        if x or y:
            await page.mouse.move(int(x), int(y))
        dy = magnitude if direction == "down" else -magnitude
        await page.mouse.wheel(0, dy)
        return await self._capture_state()

    async def wait(
        self,
        seconds: int = 5,
        duration: int = 5,
        duration_seconds: int = 5,
    ) -> ComputerState:
        sec = (
            seconds
            if seconds != 5
            else (duration if duration != 5 else duration_seconds)
        )
        sec = max(0, min(int(sec), MAX_WAIT_SECONDS))
        await asyncio.sleep(sec)
        return await self._capture_state()

    async def go_back(self) -> ComputerState:
        page = await self._get_page()
        await page.go_back()
        return await self._capture_state()

    async def go_forward(self) -> ComputerState:
        page = await self._get_page()
        await page.go_forward()
        return await self._capture_state()

    async def search(
        self,
        query: str = "",
        text: str = "",
        search_query: str = "",
        q: str = "",
    ) -> ComputerState:
        page = await self._get_page()
        raw_query = query or text or search_query or q
        search_str = raw_query.strip()
        if search_str:
            target_url = (
                f"https://www.google.com/search?q={quote_plus(search_str)}"
            )
            await page.goto(target_url)
        else:
            await page.goto(_format_url(self._start_url))
        return await self._capture_state()

    async def navigate(self, url: str = START_URL) -> ComputerState:
        page = await self._get_page()
        await page.goto(_format_url(url or self._start_url))
        return await self._capture_state()

    async def key_combination(
        self,
        keys: list[str] | str | None = None,
        key: str | None = None,
        hotkey: str | None = None,
    ) -> ComputerState:
        page = await self._get_page()
        k = key or hotkey
        if k:
            key_list = [k]
        elif isinstance(keys, str):
            key_list = [keys]
        elif keys:
            key_list = list(keys)
        else:
            key_list = ["Enter"]
        await page.keyboard.press("+".join(key_list))
        return await self._capture_state()

    async def drag_and_drop(
        self,
        x: int = 0,
        y: int = 0,
        destination_x: int = 0,
        destination_y: int = 0,
        start: list[int] | None = None,
        end: list[int] | None = None,
    ) -> ComputerState:
        page = await self._get_page()
        if start and len(start) >= 2:
            x, y = start[0], start[1]
        if end and len(end) >= 2:
            destination_x, destination_y = end[0], end[1]
        await page.mouse.move(int(x), int(y))
        await page.mouse.down()
        await page.mouse.move(int(destination_x), int(destination_y))
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
