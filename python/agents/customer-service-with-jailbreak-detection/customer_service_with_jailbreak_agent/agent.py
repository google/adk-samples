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
# limitations under the License.§

"""Agent module for the customer service agent."""

from google.adk.agents.llm_agent import Agent
from google.adk.tools.agent_tool import AgentTool
import logging
import warnings

from .config import Config
from . import prompt
from .shared_libraries.callbacks import (
    rate_limit_callback,
    before_agent,
    before_tool,
)
from .tools.tools import (
    send_call_companion_link,
    approve_discount,
    sync_ask_for_approval,
    update_salesforce_crm,
    access_cart_information,
    modify_cart,
    get_product_recommendations,
    check_product_availability,
    schedule_planting_service,
    get_available_planting_times,
    send_care_instructions,
    generate_qr_code,
)
from .sub_agents.jailbreak.agent import jailbreak_classification_agent

warnings.filterwarnings("ignore", category=UserWarning, module=".*pydantic.*")

configs = Config()

logger = logging.getLogger(__name__)


customer_service_agent = Agent(
    model=configs.agent_settings.model,
    global_instruction=prompt.GLOBAL_INSTRUCTION,
    instruction=prompt.INSTRUCTION,
    name=configs.agent_settings.name,
    tools=[
        send_call_companion_link,
        approve_discount,
        sync_ask_for_approval,
        update_salesforce_crm,
        access_cart_information,
        modify_cart,
        get_product_recommendations,
        check_product_availability,
        schedule_planting_service,
        get_available_planting_times,
        send_care_instructions,
        generate_qr_code,
        AgentTool(agent=jailbreak_classification_agent),
    ],
    before_tool_callback=before_tool,
    before_agent_callback=before_agent,
    before_model_callback=rate_limit_callback
    # TODO(b/322001639): Add a mechanism to enforce the jailbreak agent to be called before proceeding.
)


root_agent = customer_service_agent
