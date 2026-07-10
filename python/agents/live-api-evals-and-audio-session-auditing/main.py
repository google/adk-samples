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


"""FastAPI application for the info_gather_agent utilizing ADK Gemini Live API Toolkit with WebSockets and a local FileArtifactService."""

import asyncio
import base64
import json
import logging
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.agents.run_config import RunConfig
from google.adk.artifacts import FileArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Add current directory to sys.path to enable loading info_gather_agent
current_dir = Path(__file__).parent.resolve()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Load environment variables from .env file BEFORE importing agent
load_dotenv(current_dir / "info_gather_agent" / ".env")

# Import agent after loading environment variables
# pylint: disable=wrong-import-position
from info_gather_agent.agent import root_agent as agent  # noqa: E402
from utils.audio_utils import convert_artifacts_to_wav  # noqa: E402
from utils.config import LIVE_AGENT_RUN_CONFIG  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress Pydantic serialization warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Application name constant
APP_NAME = "info_gather_agent"

# ========================================
# Phase 1: Application Initialization (once at startup)
# ========================================

active_sessions = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    yield
    # Shutdown logic
    if not active_sessions:
        return

    try:
        current_dir = Path(__file__).parent.resolve()

        for s_id in list(active_sessions):
            print("\n" + "=" * 50)
            print(
                f"Server shutting down. Converting artifacts for active session {s_id} to WAV format..."
            )
            print("=" * 50)
            output_filename = f"{s_id}.wav"
            convert_artifacts_to_wav(
                artifacts_dir=str(current_dir / "artifacts"),
                output_filename=str(current_dir / output_filename),
                session_id=s_id,
            )
            active_sessions.remove(s_id)
    except Exception as e:
        logger.error(
            f"Error during shutdown audio conversion: {e}", exc_info=True
        )


app = FastAPI(lifespan=lifespan)

# Mount static files
static_dir = Path(__file__).parent / "utils" / "frontend"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Define your session service
session_service = InMemorySessionService()

# Define your artifact service to save to a local "./artifacts" folder within the blog directory
artifact_service = FileArtifactService(root_dir="./artifacts")

# Define your runner
runner = Runner(
    app_name=APP_NAME,
    agent=agent,
    session_service=session_service,
    artifact_service=artifact_service,
)

# ========================================
# HTTP Endpoints
# ========================================


@app.get("/")
async def root():
    """Serve the index.html page."""
    return FileResponse(
        Path(__file__).parent / "utils" / "frontend" / "index.html"
    )


def _write_transcript(session_id: str, speaker: str, text: str) -> None:
    """Writes a line of conversation transcript to a local text file."""
    transcripts_dir = Path(__file__).parent / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    with open(
        transcripts_dir / f"{session_id}.txt", "a", encoding="utf-8"
    ) as f:
        f.write(f"{speaker}: {text}\n")


async def handle_upstream_audio_and_text(
    websocket: WebSocket, live_request_queue: LiveRequestQueue
) -> None:
    """Receives messages from WebSocket and sends to LiveRequestQueue."""
    logger.debug("handle_upstream_audio_and_text started")
    while True:
        # Receive message from WebSocket (text or binary)
        message = await websocket.receive()

        # Handle binary frames (audio data)
        if "bytes" in message:
            audio_data = message["bytes"]
            logger.debug(
                f"Received binary audio chunk: {len(audio_data)} bytes"
            )

            audio_blob = types.Blob(
                mime_type="audio/l16;rate=16000", data=audio_data
            )
            live_request_queue.send_realtime(audio_blob)

        # Handle text frames (JSON messages)
        elif "text" in message:
            text_data = message["text"]
            logger.debug(f"Received text message: {text_data[:100]}...")

            json_message = json.loads(text_data)

            # Extract text from JSON and send to LiveRequestQueue
            if json_message.get("type") == "text":
                logger.debug(f"Sending text content: {json_message['text']}")
                content = types.Content(
                    parts=[types.Part(text=json_message["text"])]
                )
                live_request_queue.send_content(content)

            # Handle image data
            elif json_message.get("type") == "image":
                logger.debug("Received image data")

                # Decode base64 image data
                image_data = base64.b64decode(json_message["data"])
                mime_type = json_message.get("mimeType", "image/jpeg")

                logger.debug(
                    f"Sending image: {len(image_data)} bytes, type: {mime_type}"
                )

                # Send image as blob
                image_blob = types.Blob(mime_type=mime_type, data=image_data)
                live_request_queue.send_realtime(image_blob)


def _handle_input_transcription(session_id: str, transcription) -> None:
    if transcription and transcription.text:
        logger.info(
            f"[INPUT TRANSCRIPTION] [Session: {session_id}] [Finished: {transcription.finished}] {transcription.text}"
        )
        if transcription.finished:
            _write_transcript(session_id, "USER", transcription.text)


def _handle_output_transcription(session_id: str, transcription) -> None:
    if transcription and transcription.text:
        logger.info(
            f"[OUTPUT TRANSCRIPTION] [Session: {session_id}] [Finished: {transcription.finished}] {transcription.text}"
        )
        if transcription.finished:
            _write_transcript(session_id, "MODEL", transcription.text)


async def handle_downstream_events(
    websocket: WebSocket,
    runner: Runner,
    live_request_queue: LiveRequestQueue,
    user_id: str,
    session_id: str,
    run_config: RunConfig,
) -> None:
    """Receives Events from run_live() and sends to WebSocket."""
    logger.debug("handle_downstream_events started, calling runner.run_live()")
    logger.debug(
        f"Starting run_live with user_id={user_id}, session_id={session_id}"
    )
    async for event in runner.run_live(
        user_id=user_id,
        session_id=session_id,
        live_request_queue=live_request_queue,
        run_config=run_config,
    ):
        event_json = event.model_dump_json(exclude_none=True, by_alias=True)
        logger.debug(f"[SERVER] Event: {event_json}")

        if hasattr(event, "input_transcription") and event.input_transcription:
            _handle_input_transcription(session_id, event.input_transcription)

        if (
            hasattr(event, "output_transcription")
            and event.output_transcription
        ):
            _handle_output_transcription(session_id, event.output_transcription)

        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    _write_transcript(session_id, "MODEL", part.text)

        await websocket.send_text(event_json)
    logger.debug("run_live() generator completed")


# ========================================
# WebSocket Endpoint
# ========================================


@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: str,
    session_id: str,
    proactivity: bool = False,
    affective_dialog: bool = False,
) -> None:
    """WebSocket endpoint for bidirectional streaming with ADK.

    Args:
        websocket: The WebSocket connection
        user_id: User identifier
        session_id: Session identifier
        proactivity: Enable proactive audio (native audio models only)
        affective_dialog: Enable affective dialog (native audio models only)
    """
    logger.debug(
        f"WebSocket connection request: user_id={user_id}, session_id={session_id}, "
        f"proactivity={proactivity}, affective_dialog={affective_dialog}"
    )
    await websocket.accept()
    logger.debug("WebSocket connection accepted")

    # ========================================
    # Phase 2: Session Initialization (once per streaming session)
    # ========================================

    model_name = agent.model
    # Load RunConfig from central configuration repository
    run_config = LIVE_AGENT_RUN_CONFIG
    logger.debug(
        f"Native audio model detected: {model_name}, "
        f"using AUDIO response modality, "
        f"proactivity={proactivity}, affective_dialog={affective_dialog}"
    )

    # Get or create session (handles both new sessions and reconnections)
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    if not session:
        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )

    active_sessions.add(session_id)
    live_request_queue = LiveRequestQueue()

    # Run both tasks concurrently
    # Exceptions from either task will propagate and cancel the other task
    try:
        logger.debug(
            "Starting asyncio.gather for upstream and downstream tasks"
        )
        await asyncio.gather(
            handle_upstream_audio_and_text(websocket, live_request_queue),
            handle_downstream_events(
                websocket=websocket,
                runner=runner,
                live_request_queue=live_request_queue,
                user_id=user_id,
                session_id=session_id,
                run_config=run_config,
            ),
        )
        logger.debug("asyncio.gather completed normally")
    except WebSocketDisconnect:
        logger.debug("Client disconnected normally")
    except Exception as e:
        logger.error(f"Unexpected error in streaming tasks: {e}", exc_info=True)
    finally:
        # ========================================
        # Phase 4: Session Termination
        # ========================================

        logger.debug("Closing live_request_queue")
        live_request_queue.close()

        # Convert artifacts to WAV on disconnect
        try:
            print("\n" + "=" * 50)
            print(
                f"Session {session_id} ended. Converting artifacts to WAV format..."
            )
            print("=" * 50)
            current_dir = Path(__file__).parent.resolve()
            output_filename = f"{session_id}.wav"

            convert_artifacts_to_wav(
                artifacts_dir=str(current_dir / "artifacts"),
                output_filename=str(current_dir / output_filename),
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error during audio conversion: {e}", exc_info=True)
        finally:
            if session_id in active_sessions:
                active_sessions.remove(session_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
