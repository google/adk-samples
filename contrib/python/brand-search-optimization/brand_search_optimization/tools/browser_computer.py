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

"""Defines Computer Use browser controller and toolset for visual search navigation."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import urllib.parse
from typing import Any, Literal
from urllib.parse import urlparse

from google.adk.tools.computer_use.base_computer import (
    BaseComputer,
    ComputerEnvironment,
    ComputerState,
)
from google.adk.tools.computer_use.computer_use_toolset import (
    ComputerUseToolset,
)

from ..shared_libraries import constants

logger = logging.getLogger(__name__)

# Default standard viewport resolution for retail browser search
DEFAULT_SCREEN_SIZE: tuple[int, int] = (1280, 800)
DEFAULT_TIMEOUT_MS: int = 5000
DEFAULT_SCROLL_OFFSET: int = 500
MAX_WAIT_SECONDS: int = 5
POST_SCROLL_WAIT_SECONDS: float = 0.3
GOOGLE_SHOPPING_SEARCH_URL: str = "https://www.google.com/search?tbm=shop"
DEFAULT_INITIAL_SEARCH_URL: str = (
    f"{GOOGLE_SHOPPING_SEARCH_URL}&q=running+shoes"
)

_ALLOWED_URL_SCHEMES: frozenset[str] = frozenset({"http", "https"})
_DISALLOWED_HOSTS: frozenset[str] = frozenset(
    {"localhost", "127.0.0.1", "metadata.google.internal", "instance-data"}
)


def _validate_navigation_url(url: str) -> bool:
    """Validates navigation URLs to prevent SSRF against internal/metadata endpoints."""
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme.lower() not in _ALLOWED_URL_SCHEMES:
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        hostname_lower = hostname.lower()
        if hostname_lower in _DISALLOWED_HOSTS or hostname_lower.endswith(
            (".internal", ".local")
        ):
            return False
        try:
            ip = ipaddress.ip_address(hostname_lower)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
            ):
                return False
        except ValueError:
            # Hostname: resolve DNS and verify resolved IP addresses against private subnets
            try:
                addr_info = socket.getaddrinfo(
                    hostname_lower, None, type=socket.SOCK_STREAM
                )
                for item in addr_info:
                    sockaddr = item[4]
                    ip_str = sockaddr[0]
                    ip = ipaddress.ip_address(ip_str)
                    if (
                        ip.is_private
                        or ip.is_loopback
                        or ip.is_link_local
                        or ip.is_reserved
                    ):
                        return False
            except (socket.gaierror, OSError, ValueError):
                pass
        return True
    except Exception:
        return False


def _calculate_scroll_deltas(
    direction: Literal["up", "down", "left", "right"], magnitude: int
) -> tuple[int, int]:
    """Maps directional scroll names to (delta_x, delta_y) offsets."""
    if direction == "up":
        return 0, -magnitude
    elif direction == "down":
        return 0, magnitude
    elif direction == "left":
        return -magnitude, 0
    elif direction == "right":
        return magnitude, 0
    return 0, 0


def _format_url(url: str) -> str:
    """Validate and normalize an HTTP(S) URL."""
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
        parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
    return url


class MockBrowserComputer(BaseComputer):
    """Deterministic mock browser environment for offline testing and CI execution."""

    # 1x1 transparent PNG image bytes
    MOCK_SCREENSHOT_BYTES: bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
        b"\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf"
        b"\xa4q\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    def __init__(
        self,
        screen_size: tuple[int, int] = DEFAULT_SCREEN_SIZE,
        initial_url: str = DEFAULT_INITIAL_SEARCH_URL,
    ) -> None:
        self._screen_size = screen_size
        self._url = _format_url(initial_url)
        self._history: list[str] = [self._url]
        self._history_idx: int = 0

    def _update_url(self, url: str) -> None:
        """Updates the current URL and appends it to navigation history."""
        self._url = url
        self._history.append(self._url)
        self._history_idx = len(self._history) - 1

    async def screen_size(self) -> tuple[int, int]:
        return self._screen_size

    async def environment(self) -> ComputerEnvironment:
        return ComputerEnvironment.ENVIRONMENT_BROWSER

    async def open_web_browser(self) -> ComputerState:
        self._url = _format_url(self._url or DEFAULT_INITIAL_SEARCH_URL)
        return await self.current_state()

    async def click_at(self, x: int, y: int) -> ComputerState:
        return await self.current_state()

    async def hover_at(self, x: int, y: int) -> ComputerState:
        return await self.current_state()

    async def type_text_at(
        self,
        x: int,
        y: int,
        text: str,
        press_enter: bool = True,
        clear_before_typing: bool = True,
    ) -> ComputerState:
        clean_query = text.strip().replace(" ", "+")
        self._update_url(
            _format_url(f"{GOOGLE_SHOPPING_SEARCH_URL}&q={clean_query}")
        )
        return await self.current_state()

    async def scroll_document(
        self, direction: Literal["up", "down", "left", "right"]
    ) -> ComputerState:
        return await self.current_state()

    async def scroll_at(
        self,
        x: int,
        y: int,
        direction: Literal["up", "down", "left", "right"],
        magnitude: int,
    ) -> ComputerState:
        return await self.current_state()

    async def wait(self, seconds: int) -> ComputerState:
        return await self.current_state()

    async def go_back(self) -> ComputerState:
        if self._history_idx > 0:
            self._history_idx -= 1
            self._url = self._history[self._history_idx]
        return await self.current_state()

    async def go_forward(self) -> ComputerState:
        if self._history_idx < len(self._history) - 1:
            self._history_idx += 1
            self._url = self._history[self._history_idx]
        return await self.current_state()

    async def search(self) -> ComputerState:
        self._update_url(_format_url(GOOGLE_SHOPPING_SEARCH_URL))
        return await self.current_state()

    async def navigate(self, url: str) -> ComputerState:
        formatted_url = _format_url(url)
        if not _validate_navigation_url(formatted_url):
            logger.warning(
                "Rejected navigation to disallowed URL: %s", formatted_url
            )
            return await self.current_state()
        self._update_url(formatted_url)
        return await self.current_state()

    async def key_combination(self, keys: list[str]) -> ComputerState:
        return await self.current_state()

    async def drag_and_drop(
        self, x: int, y: int, destination_x: int, destination_y: int
    ) -> ComputerState:
        return await self.current_state()

    async def current_state(self) -> ComputerState:
        return ComputerState(
            screenshot=self.MOCK_SCREENSHOT_BYTES,
            url=self._url,
        )


class PlaywrightBrowserComputer(BaseComputer):
    """Controls a browser session using Playwright for Gemini Computer Use."""

    def __init__(
        self,
        screen_size: tuple[int, int] = DEFAULT_SCREEN_SIZE,
        headless: bool = True,
    ) -> None:
        self._screen_size = screen_size
        self._headless = headless
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._contexts: dict[str, Any] = {}
        self._pages: dict[str, Any] = {}
        self._current_session_id: str = "default"

    async def prepare(self, tool_context: Any) -> None:
        """Binds active session context to avoid multi-session page contention."""
        if (
            tool_context
            and hasattr(tool_context, "session")
            and tool_context.session
        ):
            self._current_session_id = str(tool_context.session.id)
        else:
            self._current_session_id = "default"

    @property
    def _page(self) -> Any:
        return self._pages.get(self._current_session_id)

    async def screen_size(self) -> tuple[int, int]:
        return self._screen_size

    async def environment(self) -> ComputerEnvironment:
        return ComputerEnvironment.ENVIRONMENT_BROWSER

    async def _ensure_browser(self) -> None:
        sid = self._current_session_id
        if sid not in self._pages or self._pages[sid] is None:
            try:
                from playwright.async_api import async_playwright

                if self._playwright is None:
                    self._playwright = await async_playwright().start()
                if self._browser is None:
                    self._browser = await self._playwright.chromium.launch(
                        headless=self._headless,
                    )
                if sid not in self._contexts or self._contexts[sid] is None:
                    self._contexts[sid] = await self._browser.new_context(
                        viewport={
                            "width": self._screen_size[0],
                            "height": self._screen_size[1],
                        }
                    )
                self._pages[sid] = await self._contexts[sid].new_page()
            except Exception as e:
                logger.warning(
                    "Playwright initialization failed (%s); falling back to dummy state",
                    e,
                )

    async def open_web_browser(self) -> ComputerState:
        await self._ensure_browser()
        if self._page and (
            not self._page.url or self._page.url == "about:blank"
        ):
            target_url = _format_url("https://www.google.com")
            await self._page.goto(target_url, wait_until="domcontentloaded")
        return await self.current_state()

    async def click_at(self, x: int, y: int) -> ComputerState:
        await self._ensure_browser()
        if self._page:
            await self._page.mouse.click(x, y)
            try:
                await self._page.wait_for_load_state(
                    "domcontentloaded", timeout=DEFAULT_TIMEOUT_MS
                )
            except Exception:
                pass
        return await self.current_state()

    async def hover_at(self, x: int, y: int) -> ComputerState:
        await self._ensure_browser()
        if self._page:
            await self._page.mouse.move(x, y)
        return await self.current_state()

    async def type_text_at(
        self,
        x: int,
        y: int,
        text: str,
        press_enter: bool = True,
        clear_before_typing: bool = True,
    ) -> ComputerState:
        await self._ensure_browser()
        if self._page:
            await self._page.mouse.click(x, y)
            if clear_before_typing:
                await self._page.keyboard.press("Control+A")
                await self._page.keyboard.press("Backspace")
            await self._page.keyboard.type(text)
            if press_enter:
                await self._page.keyboard.press("Enter")
                try:
                    await self._page.wait_for_load_state(
                        "domcontentloaded", timeout=DEFAULT_TIMEOUT_MS
                    )
                except Exception:
                    pass
        return await self.current_state()

    async def scroll_document(
        self, direction: Literal["up", "down", "left", "right"]
    ) -> ComputerState:
        await self._ensure_browser()
        if self._page:
            delta_x, delta_y = _calculate_scroll_deltas(
                direction, DEFAULT_SCROLL_OFFSET
            )
            await self._page.mouse.wheel(delta_x, delta_y)
            await asyncio.sleep(POST_SCROLL_WAIT_SECONDS)
        return await self.current_state()

    async def scroll_at(
        self,
        x: int,
        y: int,
        direction: Literal["up", "down", "left", "right"],
        magnitude: int,
    ) -> ComputerState:
        await self._ensure_browser()
        if self._page:
            await self._page.mouse.move(x, y)
            delta_x, delta_y = _calculate_scroll_deltas(direction, magnitude)
            await self._page.mouse.wheel(delta_x, delta_y)
            await asyncio.sleep(POST_SCROLL_WAIT_SECONDS)
        return await self.current_state()

    async def wait(self, seconds: int) -> ComputerState:
        await self._ensure_browser()
        await asyncio.sleep(min(seconds, MAX_WAIT_SECONDS))
        return await self.current_state()

    async def go_back(self) -> ComputerState:
        await self._ensure_browser()
        if self._page:
            try:
                await self._page.go_back(timeout=DEFAULT_TIMEOUT_MS)
            except Exception:
                pass
        return await self.current_state()

    async def go_forward(self) -> ComputerState:
        await self._ensure_browser()
        if self._page:
            try:
                await self._page.go_forward(timeout=DEFAULT_TIMEOUT_MS)
            except Exception:
                pass
        return await self.current_state()

    async def search(self) -> ComputerState:
        await self._ensure_browser()
        if self._page:
            target_url = _format_url(GOOGLE_SHOPPING_SEARCH_URL)
            await self._page.goto(
                target_url,
                wait_until="domcontentloaded",
            )
        return await self.current_state()

    async def navigate(self, url: str) -> ComputerState:
        formatted_url = _format_url(url)
        if not _validate_navigation_url(formatted_url):
            logger.warning(
                "Rejected navigation to disallowed URL: %s", formatted_url
            )
            return await self.current_state()
        await self._ensure_browser()
        if self._page:
            await self._page.goto(formatted_url, wait_until="domcontentloaded")
        return await self.current_state()

    async def key_combination(self, keys: list[str]) -> ComputerState:
        await self._ensure_browser()
        if self._page and keys:
            await self._page.keyboard.press("+".join(keys))
        return await self.current_state()

    async def drag_and_drop(
        self, x: int, y: int, destination_x: int, destination_y: int
    ) -> ComputerState:
        await self._ensure_browser()
        if self._page:
            await self._page.mouse.move(x, y)
            await self._page.mouse.down()
            await self._page.mouse.move(destination_x, destination_y)
            await self._page.mouse.up()
        return await self.current_state()

    async def current_state(self) -> ComputerState:
        await self._ensure_browser()
        if self._page:
            try:
                screenshot_bytes = await self._page.screenshot(type="png")
                return ComputerState(
                    screenshot=screenshot_bytes,
                    url=self._page.url or "about:blank",
                )
            except Exception as e:
                logger.warning(
                    "Screenshot capture failed (%s); using fallback bytes", e
                )
        return ComputerState(
            screenshot=MockBrowserComputer.MOCK_SCREENSHOT_BYTES,
            url="about:blank",
        )

    async def close(self) -> None:
        for page in self._pages.values():
            if page is not None:
                try:
                    await page.close()
                except Exception:
                    pass
        self._pages.clear()
        for context in self._contexts.values():
            if context is not None:
                try:
                    await context.close()
                except Exception:
                    pass
        self._contexts.clear()
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None


def get_browser_computer() -> BaseComputer:
    """Returns PlaywrightBrowserComputer if enabled, otherwise MockBrowserComputer."""
    if constants.DISABLE_WEB_DRIVER:
        return MockBrowserComputer()
    try:
        import playwright  # noqa: F401

        return PlaywrightBrowserComputer()
    except ImportError:
        logger.info(
            "Playwright is not installed. Using MockBrowserComputer for offline mode."
        )
        return MockBrowserComputer()


def get_computer_use_toolset() -> ComputerUseToolset:
    """Creates a ComputerUseToolset wrapping the configured browser computer."""
    computer = get_browser_computer()
    return ComputerUseToolset(computer=computer)
