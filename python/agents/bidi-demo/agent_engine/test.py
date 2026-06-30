"""
Test the deployed bidi-demo agent with bidirectional streaming.

Requires the agent-engine optional dependencies:
    uv sync --extra agent-engine

Usage:
    uv run agent_engine/test.py                # automated test
    uv run agent_engine/test.py --interactive  # interactive chat
"""

import argparse
import asyncio
import os
import time

import numpy as np
import vertexai
from dotenv import load_dotenv
from google.adk.agents.live_request_queue import LiveRequest
from google.adk.events import Event
from google.genai import types

load_dotenv(
    os.path.join(os.path.dirname(__file__), "..", "app", ".env"), override=True
)

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-east1")


def get_resource_name() -> str:
    with open("agent_resource_name.txt") as f:
        return f.read().strip()


MAX_EVENTS = 50


async def test_automated(client, resource_name: str):  # noqa: PLR0912
    """Run automated test queries against the deployed agent."""

    def prepare_live_request(text: str) -> LiveRequest:
        part = types.Part.from_text(text=text)
        content = types.Content(parts=[part])
        return LiveRequest(content=content)

    queries = [
        "What is the population of Tokyo?",
        "Who won the last FIFA World Cup?",
    ]

    async with client.aio.live.agent_engines.connect(
        agent_engine=resource_name,
        config={"class_method": "bidi_stream_query"},
    ) as connection:
        user_id = "test_user"
        first_req = True

        for query in queries:
            print(f"\n{'=' * 50}")
            print(f"Query: {query}")
            print("=" * 50)

            if first_req:
                await connection.send(
                    {
                        "user_id": user_id,
                        "live_request": prepare_live_request(query).dict(),
                    }
                )
                first_req = False
            else:
                await connection.send(prepare_live_request(query).dict())

            audio_chunks = []
            event_count = 0
            while True:
                received = await connection.receive()
                event_data = received.get("bidiStreamOutput", received)
                event_count += 1

                print(
                    f"  Event {event_count}: keys="
                    f"{list(event_data.keys()) if isinstance(event_data, dict) else type(event_data)}"
                )

                if isinstance(event_data, dict) and "output" in event_data:
                    output = event_data["output"]
                    if output == "end of turn":
                        print("  [end of turn]")
                        break
                    print(f"  Output: {output}")
                    continue

                if isinstance(event_data, dict) and "actions" in event_data:
                    event_data.pop("requested_tool_confirmations", None)
                    if "requested_tool_confirmations" in event_data.get(
                        "actions", {}
                    ):
                        del event_data["actions"][
                            "requested_tool_confirmations"
                        ]

                try:
                    event = Event.model_validate(event_data)
                except Exception as e:
                    print(f"  (raw data, can't parse as Event: {str(e)[:100]})")
                    continue

                part = (
                    event.content
                    and event.content.parts
                    and event.content.parts[0]
                )

                if not part and audio_chunks:
                    break
                if not part:
                    continue

                if part.inline_data and part.inline_data.data:
                    data = np.frombuffer(part.inline_data.data, dtype=np.int16)
                    audio_chunks.append(data)
                elif part.text:
                    print(f"  Response: {part.text}")
                elif part.function_call:
                    print(
                        f"  Tool call: {part.function_call.name}({part.function_call.args})"
                    )

                if event_count > MAX_EVENTS:
                    print("  (max events reached)")
                    break

            if audio_chunks:
                full_audio = np.concatenate(audio_chunks)
                print(
                    f"  Received audio: {len(full_audio)} samples "
                    f"({len(full_audio) / 24000:.1f}s at 24kHz)"
                )

    print("\nAll tests passed!")


async def test_interactive(client, resource_name: str):
    """Run interactive chat with the deployed agent."""

    def prepare_live_request(text: str) -> LiveRequest:
        part = types.Part.from_text(text=text)
        content = types.Content(parts=[part])
        return LiveRequest(content=content)

    async with client.aio.live.agent_engines.connect(
        agent_engine=resource_name,
        config={"class_method": "bidi_stream_query"},
    ) as connection:
        print("Assistant Ready! (type 'exit' to quit)\n")
        print("Example questions:")
        print("  - What is the population of Tokyo?")
        print("  - Who won the last FIFA World Cup?")
        print()

        user_id = "interactive_user"
        first_req = True

        while True:
            time.sleep(0.1)
            input_text = input("You: ")
            if input_text.lower() in ("exit", "quit"):
                break

            if first_req:
                await connection.send(
                    {
                        "user_id": user_id,
                        "live_request": prepare_live_request(input_text).dict(),
                    }
                )
                first_req = False
            else:
                await connection.send(prepare_live_request(input_text).dict())

            audio_data = []
            seen_audio = False
            print("Assistant: ", end="", flush=True)

            while True:
                received = await connection.receive()
                event_data = received["bidiStreamOutput"]

                if (
                    "actions" in event_data
                    and "requested_tool_confirmations"
                    in event_data.get("actions", {})
                ):
                    del event_data["actions"]["requested_tool_confirmations"]

                event = Event.model_validate(event_data)
                part = (
                    event.content
                    and event.content.parts
                    and event.content.parts[0]
                )

                if not part and seen_audio:
                    break
                if not part:
                    break

                if part.inline_data and part.inline_data.data:
                    seen_audio = True
                    data = np.frombuffer(part.inline_data.data, dtype=np.int16)
                    audio_data.append(data)
                elif part.text:
                    print(part.text)
                    break
                else:
                    break

            if audio_data:
                full_audio = np.concatenate(audio_data)
                print(f"[audio: {len(full_audio) / 24000:.1f}s]")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Test the deployed bidi-demo agent"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Run interactive chat"
    )
    args = parser.parse_args()

    vertexai.init(project=PROJECT_ID, location=LOCATION)
    client = vertexai.Client(project=PROJECT_ID, location=LOCATION)

    resource_name = get_resource_name()
    print(f"Testing agent: {resource_name}")

    if args.interactive:
        asyncio.run(test_interactive(client, resource_name))
    else:
        asyncio.run(test_automated(client, resource_name))


if __name__ == "__main__":
    main()
