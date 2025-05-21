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

import time
import warnings

import selenium
from google.adk.agents.llm_agent import Agent
from google.adk.tools.load_artifacts_tool import load_artifacts_tool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from PIL import Image
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

from ...shared_libraries import constants
from . import prompt

warnings.filterwarnings("ignore", category=UserWarning)

if not constants.DISABLE_WEB_DRIVER:
    options = Options()
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--verbose")
    options.add_argument("user-data-dir=/tmp/selenium")

    driver = selenium.webdriver.Chrome(options=options)


def go_to_url(url: str) -> str:
    """Navigates the browser to the given URL."""
    print(f"🌐 Navigating to URL: {url}")  # Added print statement
    driver.get(url.strip())
    return f"Navigated to URL: {url}"


async def take_screenshot(tool_context: ToolContext) -> dict:
    """Takes a screenshot and saves it with the given filename. called 'load artifacts' after to load the image"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    print(f"📸 Taking screenshot and saving as: {filename}")
    driver.save_screenshot(filename)

    image = Image.open(filename)

    await tool_context.save_artifact(
        filename,
        types.Part.from_bytes(data=image.tobytes(), mime_type="image/png"),
    )

    return {"status": "ok", "filename": filename}


def click_at_coordinates(x: int, y: int) -> str:
    """Clicks at the specified coordinates on the screen."""
    driver.execute_script(f"window.scrollTo({x}, {y});")
    driver.find_element(By.TAG_NAME, "body").click()


def find_element_with_text(text: str) -> str:
    """Finds an element on the page with the given text."""
    print(f"🔍 Finding element with text: '{text}'")  # Added print statement

    try:
        element = driver.find_element(By.XPATH, f"//*[text()='{text}']")
        if element:
            return "Element found."
        else:
            return "Element not found."
    except selenium.common.exceptions.NoSuchElementException:
        return "Element not found."
    except selenium.common.exceptions.ElementNotInteractableException:
        return "Element not interactable, cannot click."


def click_element_with_text(text: str) -> str:
    """Clicks on an element on the page with the given text."""
    print(f"🖱️ Clicking element with text: '{text}'")  # Added print statement

    try:
        element = driver.find_element(By.XPATH, f"//*[text()='{text}']")
        element.click()
        return f"Clicked element with text: {text}"
    except selenium.common.exceptions.NoSuchElementException:
        return "Element not found, cannot click."
    except selenium.common.exceptions.ElementNotInteractableException:
        return "Element not interactable, cannot click."
    except selenium.common.exceptions.ElementClickInterceptedException:
        return "Element click intercepted, cannot click."


def enter_text_into_element(text_to_enter: str, element_id: str) -> str:
    """Enters text into an element with the given ID."""
    print(
        f"📝 Entering text '{text_to_enter}' into element with ID: {element_id}"
    )  # Added print statement

    try:
        input_element = driver.find_element(By.ID, element_id)
        input_element.send_keys(text_to_enter)
        return (
            f"Entered text '{text_to_enter}' into element with ID: {element_id}"
        )
    except selenium.common.exceptions.NoSuchElementException:
        return "Element with given ID not found."
    except selenium.common.exceptions.ElementNotInteractableException:
        return "Element not interactable, cannot click."


def scroll_down_screen() -> str:
    """Scrolls down the screen by a moderate amount."""
    print("⬇️ scroll the screen")  # Added print statement
    driver.execute_script("window.scrollBy(0, 500)")
    return "Scrolled down the screen."


def get_page_source() -> str:
    LIMIT = 1000000
    """Returns the current page source."""
    print("📄 Getting page source...")  # Added print statement
    return driver.page_source[0:LIMIT]


def analyze_webpage_and_determine_action(
    page_source: str, user_task: str, tool_context: ToolContext
) -> str:
    """Analyzes the webpage and determines the next action (scroll, click, etc.)."""
    print(
        "🤔 Analyzing webpage and determining next action..."
    )  # Added print statement

    analysis_prompt = f"""
    You are an expert web page analyzer.
    You have been tasked with controlling a web browser to achieve a user's goal.
    The user's task is: {user_task}
    Here is the current HTML source code of the webpage:
    ```html
    {page_source}
    ```

    Based on the webpage content and the user's task, determine the next best action to take.
    Consider actions like: completing page source, scrolling down to see more content, clicking on links or buttons to navigate, or entering text into input fields.

    Think step-by-step:
    1. Briefly analyze the user's task and the webpage content.
    2. If source code appears to be incomplete, complete it to make it valid html. Keep the product titles as is. Only complete missing html syntax
    3. Identify potential interactive elements on the page (links, buttons, input fields, etc.).
    4. Determine if scrolling is necessary to reveal more content.
    5. Decide on the most logical next action to progress towards completing the user's task.

    Your response should be a concise action plan, choosing from these options:
    - "COMPLETE_PAGE_SOURCE": If source code appears to be incomplete, complte it to make it valid html
    - "SCROLL_DOWN": If more content needs to be loaded by scrolling.
    - "CLICK: <element_text>": If a specific element with text <element_text> should be clicked. Replace <element_text> with the actual text of the element.
    - "ENTER_TEXT: <element_id>, <text_to_enter>": If text needs to be entered into an input field. Replace <element_id> with the ID of the input element and <text_to_enter> with the text to enter.
    - "TASK_COMPLETED": If you believe the user's task is likely completed on this page.
    - "STUCK": If you are unsure what to do next or cannot progress further.
    - "ASK_USER": If you need clarification from the user on what to do next.

    If you choose "CLICK" or "ENTER_TEXT", ensure the element text or ID is clearly identifiable from the webpage source. If multiple similar elements exist, choose the most relevant one based on the user's task.
    If you are unsure, or if none of the above actions seem appropriate, default to "ASK_USER".

    Example Responses:
    - SCROLL_DOWN
    - CLICK: Learn more
    - ENTER_TEXT: search_box_id, Gemini API
    - TASK_COMPLETED
    - STUCK
    - ASK_USER

    What is your action plan?
    """
    return analysis_prompt


search_results_agent = Agent(
    model=constants.MODEL,
    name="search_results_agent",
    description="Get top 3 search results info for a keyword using web browsing",
    instruction=prompt.SEARCH_RESULT_AGENT_PROMPT,
    tools=[
        go_to_url,
        take_screenshot,
        find_element_with_text,
        click_element_with_text,
        enter_text_into_element,
        scroll_down_screen,
        get_page_source,
        load_artifacts_tool,
        analyze_webpage_and_determine_action,
    ],
)
