#!/usr/bin/env python3
"""Voice-based product search using Gemini Live API.

Uses the google-genai SDK's Live API for bidirectional audio streaming
with tool calling to search the product catalog via Vector Search 2.0.

Usage:
    # Text mode (no microphone needed, good for testing)
    python scripts/live_search.py --project-id YOUR_PROJECT --mode text

    # Voice mode (requires microphone)
    python scripts/live_search.py --project-id YOUR_PROJECT --mode voice

    # With custom collection
    python scripts/live_search.py --project-id YOUR_PROJECT \
        --collection projects/PROJECT/locations/REGION/collections/COLLECTION

Environment variables:
    GOOGLE_GENAI_USE_VERTEXAI: Set to "True" to use Vertex AI (default)
    VECTOR_SEARCH_COLLECTION: Vector Search 2.0 collection resource name
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from google import genai
from google.genai import types
from google.cloud import vectorsearch

# Add _shared to path for shared utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "_shared"))
from setup_utils import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL = "gemini-live-2.5-flash-native-audio"

SYSTEM_INSTRUCTION = """You are a helpful product search assistant for an online store.

Your job is to help customers find products using natural conversation.

When a user describes what they're looking for:
1. Use the search_products tool to find matching products.
2. Present the top results clearly: name, price, rating, and a brief description.
3. If the query is vague, ask ONE clarifying question (price range, category, or brand).
4. Keep responses concise and conversational.

When presenting results:
- Lead with the best match.
- Mention price and rating.
- If no results found, suggest broadening the search.

Do NOT make up products. Only return results from the search tool."""

# Tool declaration for product search
SEARCH_TOOL = {
    "function_declarations": [
        {
            "name": "search_products",
            "description": (
                "Search the product catalog using a natural language query. "
                "Returns matching products with name, price, description, "
                "category, brand, rating, and stock information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query describing what the user wants",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default 5)",
                    },
                },
                "required": ["query"],
            },
        }
    ]
}


def search_products_in_collection(
    query: str,
    collection_path: str,
    max_results: int = 5,
) -> str:
    """Execute a semantic search against the Vector Search 2.0 collection.

    Returns formatted product results as a string.
    """
    try:
        client = vectorsearch.DataObjectSearchServiceClient()

        request = vectorsearch.SearchDataObjectsRequest(
            parent=collection_path,
            semantic_search=vectorsearch.SemanticSearch(
                search_text=query,
                search_field="text_embedding",
                task_type="QUESTION_ANSWERING",
                top_k=max_results,
                output_fields=vectorsearch.OutputFields(
                    data_fields=[
                        "product_id", "name", "price", "description",
                        "category", "brand", "rating", "stock",
                    ]
                ),
            ),
        )

        response = client.search_data_objects(request=request)

        results = []
        for i, result in enumerate(response, 1):
            data = result.data_object.data
            name = data.get("name", "Unknown")
            price = data.get("price", "N/A")
            description = data.get("description", "")
            category = data.get("category", "")
            brand = data.get("brand", "")
            rating = data.get("rating", "")
            stock = data.get("stock", "")

            entry = f"{i}. {name}"
            if price != "N/A":
                entry += f" - ${price}"
            if brand:
                entry += f" ({brand})"
            if rating:
                entry += f" | {rating}/5"
            try:
                if stock is not None and str(stock) != "" and int(float(str(stock))) == 0:
                    entry += " | OUT OF STOCK"
            except (ValueError, TypeError):
                pass
            if description:
                entry += f"\n   {description[:150]}"
            if category:
                entry += f"\n   Category: {category}"

            results.append(entry)

        if not results:
            return "No products found matching your query. Try broadening your search."

        return f"Found {len(results)} products:\n\n" + "\n\n".join(results)

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search encountered an error: {e}. Please try again."


class LiveProductSearch:
    """Voice or text product search using Gemini Live API."""

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        collection_path: str = "",
        voice_name: str = "Kore",
    ):
        self.project_id = project_id
        self.location = location
        self.collection_path = collection_path
        self.voice_name = voice_name

        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

        self.client = genai.Client()

    def _build_config(self, mode: str) -> dict:
        """Build LiveConnectConfig for the session."""
        config = {
            "system_instruction": SYSTEM_INSTRUCTION,
            "tools": [SEARCH_TOOL],
            "input_audio_transcription": {},
            "output_audio_transcription": {},
        }

        if mode == "voice":
            config["response_modalities"] = ["AUDIO"]
            config["speech_config"] = {
                "voice_config": {
                    "prebuilt_voice_config": {"voice_name": self.voice_name}
                }
            }
        else:
            config["response_modalities"] = ["TEXT"]

        return config

    def _handle_tool_call(self, function_call) -> str:
        """Execute a tool call and return the result."""
        if function_call.name == "search_products":
            args = dict(function_call.args) if function_call.args else {}
            query = args.get("query", "")
            max_results = args.get("max_results", 5)

            if not self.collection_path:
                return "No product collection configured. Please set --collection or VECTOR_SEARCH_COLLECTION."

            logger.info(f"Searching products: '{query}' (max {max_results})")
            return search_products_in_collection(
                query=query,
                collection_path=self.collection_path,
                max_results=int(max_results),
            )
        else:
            return f"Unknown tool: {function_call.name}"

    async def run_text_mode(self):
        """Interactive text-based session (no microphone needed)."""
        config = self._build_config("text")

        print("\n--- Product Search (Text Mode) ---")
        print("Type your query and press Enter. Type 'quit' to exit.\n")

        async with self.client.aio.live.connect(model=MODEL, config=config) as session:
            while True:
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("You: ")
                    )
                except (EOFError, KeyboardInterrupt):
                    break

                if user_input.strip().lower() in ("quit", "exit", "q"):
                    break

                if not user_input.strip():
                    continue

                await session.send_client_content(
                    turns={"role": "user", "parts": [{"text": user_input}]},
                    turn_complete=True,
                )

                full_response = ""
                async for response in session.receive():
                    # Handle tool calls
                    if response.tool_call:
                        function_responses = []
                        for fc in response.tool_call.function_calls:
                            result = self._handle_tool_call(fc)
                            function_responses.append(
                                types.FunctionResponse(
                                    id=fc.id,
                                    name=fc.name,
                                    response={"result": result},
                                )
                            )
                        await session.send_tool_response(
                            function_responses=function_responses
                        )
                        continue

                    # Handle text response
                    if response.server_content:
                        sc = response.server_content

                        if sc.model_turn:
                            for part in sc.model_turn.parts:
                                if part.text:
                                    full_response += part.text

                        if sc.turn_complete:
                            break

                if full_response:
                    print(f"\nAssistant: {full_response}\n")

        print("\nSession ended.")

    async def run_voice_mode(self):
        """Voice-based session with microphone input and audio output."""
        try:
            import pyaudio
        except ImportError:
            print("ERROR: pyaudio is required for voice mode.")
            print("Install it with: pip install pyaudio")
            print("On macOS: brew install portaudio && pip install pyaudio")
            sys.exit(1)

        config = self._build_config("voice")

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        INPUT_RATE = 16000
        OUTPUT_RATE = 24000

        audio = pyaudio.PyAudio()

        print("\n--- Product Search (Voice Mode) ---")
        print("Speak your query. Press Ctrl+C to exit.\n")

        async with self.client.aio.live.connect(model=MODEL, config=config) as session:

            audio_out_queue = asyncio.Queue()

            async def capture_audio():
                """Capture microphone audio and send to Gemini."""
                stream_in = audio.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=INPUT_RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                )
                logger.info("Microphone active. Speak your query...")

                try:
                    while True:
                        audio_chunk = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: stream_in.read(CHUNK, exception_on_overflow=False),
                        )
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=audio_chunk,
                                mime_type="audio/pcm;rate=16000",
                            )
                        )
                except asyncio.CancelledError:
                    pass
                finally:
                    stream_in.stop_stream()
                    stream_in.close()

            async def play_audio():
                """Play audio responses from the queue."""
                stream_out = audio.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=OUTPUT_RATE,
                    output=True,
                )

                try:
                    while True:
                        audio_data = await audio_out_queue.get()
                        if audio_data is None:
                            break
                        await asyncio.get_event_loop().run_in_executor(
                            None, stream_out.write, audio_data
                        )
                except asyncio.CancelledError:
                    pass
                finally:
                    stream_out.stop_stream()
                    stream_out.close()

            async def receive_responses():
                """Receive and process responses from Gemini."""
                try:
                    async for response in session.receive():
                        # Handle tool calls
                        if response.tool_call:
                            function_responses = []
                            for fc in response.tool_call.function_calls:
                                result = self._handle_tool_call(fc)
                                function_responses.append(
                                    types.FunctionResponse(
                                        id=fc.id,
                                        name=fc.name,
                                        response={"result": result},
                                    )
                                )
                            await session.send_tool_response(
                                function_responses=function_responses
                            )
                            continue

                        # Handle server content
                        if response.server_content:
                            sc = response.server_content

                            # Handle audio output
                            if sc.model_turn:
                                for part in sc.model_turn.parts:
                                    if part.inline_data:
                                        await audio_out_queue.put(part.inline_data.data)

                            # Show transcriptions
                            if sc.input_transcription and sc.input_transcription.text:
                                print(f"  You: {sc.input_transcription.text}")

                            if sc.output_transcription and sc.output_transcription.text:
                                print(f"  Assistant: {sc.output_transcription.text}")

                            # Handle interruption
                            if sc.interrupted:
                                while not audio_out_queue.empty():
                                    audio_out_queue.get_nowait()

                except asyncio.CancelledError:
                    pass

            tasks = [
                asyncio.create_task(capture_audio()),
                asyncio.create_task(play_audio()),
                asyncio.create_task(receive_responses()),
            ]

            try:
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                logger.info("Stopping...")
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                audio.terminate()

        print("\nSession ended.")


def main():
    parser = argparse.ArgumentParser(
        description="Voice-based product search using Gemini Live API"
    )
    parser.add_argument("--project-id", help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="GCP region")
    parser.add_argument(
        "--collection",
        default="",
        help="Vector Search 2.0 collection resource name",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "voice"],
        default="text",
        help="Interaction mode: text (default) or voice",
    )
    parser.add_argument(
        "--voice",
        default="Kore",
        help="Voice name for audio output (default: Kore)",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Path to design-spec.md for defaults",
    )

    args = parser.parse_args()

    # Load config defaults
    if args.config:
        cfg = load_config(args.config)
        if not args.project_id:
            args.project_id = cfg.get("gcp_project_id", "")
        if not args.collection and cfg.get("collection_id"):
            args.collection = (
                f"projects/{args.project_id}/locations/{args.location}"
                f"/collections/{cfg['collection_id']}"
            )

    # Env var fallbacks
    if not args.project_id:
        args.project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    if not args.collection:
        args.collection = os.getenv("VECTOR_SEARCH_COLLECTION", "")

    if not args.project_id:
        parser.error(
            "--project-id is required (or set GOOGLE_CLOUD_PROJECT env var)"
        )

    if not args.collection:
        logger.warning(
            "No collection specified. Tool calls will return an error. "
            "Set --collection or VECTOR_SEARCH_COLLECTION env var."
        )

    search = LiveProductSearch(
        project_id=args.project_id,
        location=args.location,
        collection_path=args.collection,
        voice_name=args.voice,
    )

    if args.mode == "voice":
        asyncio.run(search.run_voice_mode())
    else:
        asyncio.run(search.run_text_mode())


if __name__ == "__main__":
    main()
