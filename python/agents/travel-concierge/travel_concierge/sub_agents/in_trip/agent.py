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

"""In-trip agent. A post-booking agent covering the user experience during the trip."""

from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool

from travel_concierge.sub_agents.in_trip import prompt
from travel_concierge.sub_agents.in_trip.tools import (
    transit_coordination,
    flight_status_check,
    event_booking_check,
    weather_impact_check,
)

from travel_concierge.tools.memory import memorize


# This sub-agent is expected to be called every day closer to the trip, and frequently several times a day during the trip.
day_of_agent = Agent(
    model="gemini-2.0-flash",
    name="day_of_agent",
    description="Day_of agent is the agent handling the travel logistics of a trip.",
    instruction=transit_coordination,
)


trip_monitor_agent = Agent(
    model="gemini-2.0-flash",
    name="trip_monitor_agent",
    description="Monitor aspects of a itinerary and bring attention to items that necessitate changes",
    instruction=prompt.TRIP_MONITOR_INSTR,
    tools=[flight_status_check, event_booking_check, weather_impact_check],
    output_key="daily_checks",  # can be sent via email.
)


in_trip_agent = Agent(
    model="gemini-2.0-flash",
    name="in_trip_agent",
    description="Provide information about what the users need as part of the tour.",
    instruction=prompt.INTRIP_INSTR,
    sub_agents=[
        trip_monitor_agent
    ],  # This can be run as an AgentTool. Illustrate as an Agent for demo purpose.
    tools=[
        AgentTool(agent=day_of_agent), 
        memorize
    ],
)
