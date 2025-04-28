# app/live_server.py
import asyncio
import json
import logging
import os
import uuid
import base64 # Needed for decoding/encoding audio
from pathlib import Path
import sys # To modify path for imports if needed
from contextlib import suppress # For ignoring CancelledError during cleanup

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse # Use Starlette directly for FileResponse

# --- ADK Imports ---
from google.adk.runners import Runner
from google.adk.agents.run_config import RunConfig
from google.adk.sessions import InMemorySessionService, Session # Using in-memory for simplicity
from google.adk.agents.live_request_queue import LiveRequestQueue
from google.adk.events import Event, EventActions # Import EventActions for state delta
# Use alias for genai types to avoid conflicts if any
from google.genai import types as genai_types

from typing import Any, Dict, Optional


# --- Configuration ---
from dotenv import load_dotenv
# Load .env file from the parent directory (project root)
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

# Setup logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO').upper(),
                   format='%(asctime)s - %(name)s - %(levelname)s - LIVE_SERVER - %(message)s')
logger = logging.getLogger(__name__)

# --- Import the Host Agent ---
try:
    # Add project root to path to find host_agent package
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.debug(f"Added project root to sys.path: {project_root}")

    from host_agent.agent import host_agent # Now this should work
    logger.info("Successfully imported host_agent.")
except ImportError as e:
     # Be specific about the import error
     logger.critical(f"Could not import host_agent: {e}. Check sys.path and ensure host_agent/agent.py exists and is correct.", exc_info=True)
     host_agent = None
except Exception as e:
    logger.critical(f"An unexpected error occurred importing host_agent: {e}", exc_info=True)
    host_agent = None

# Ensure host_agent was loaded
if not host_agent:
    logger.critical("Host agent failed to load. Exiting.")
    sys.exit(1) # Exit if the core agent is missing

APP_NAME = "ProjectHorizonLive" # Consistent app name
STATIC_DIR = Path(__file__).parent / "static"

# --- ADK Setup ---
session_service = InMemorySessionService()
runner = Runner(
    app_name=APP_NAME,
    agent=host_agent,
    session_service=session_service,
    # artifact_service=..., # Add if needed
    # memory_service=..., # Add if needed
)
logger.info(f"ADK Runner initialized for agent: {host_agent.name}")

# --- FastAPI App ---
app = FastAPI(title="Project Horizon Live Server")

# --- WebSocket Endpoint ---
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Handles WebSocket connection for live agent interaction."""
    await websocket.accept()
    logger.info(f"WebSocket client connected for session: {session_id}")
    # Create a unique user ID based on session for this example
    user_id = f"live_user_{session_id}"

    live_events: Optional[asyncio.StreamReader] = None # Define types for clarity
    live_request_queue: Optional[LiveRequestQueue] = None
    adk_run_task: Optional[asyncio.Task] = None
    client_listener_task: Optional[asyncio.Task] = None
    session: Optional[Session] = None # Define session in the broader scope

    try:
        # --- Create or Get ADK Session ---
        # Use **keyword arguments**
        session = session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
        if not session:
            session = session_service.create_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id, state={}
            )
            logger.info(f"Created new ADK session: {session_id}")
        else:
             logger.info(f"Resumed existing ADK session: {session_id}")

        # --- Setup ADK Live Run ---
        live_request_queue = LiveRequestQueue()
        # Configure run for audio output
        run_config = RunConfig(response_modalities=["AUDIO"])
        # Start the agent's live execution loop in the background
        live_events = runner.run_live(
            session=session,
            live_request_queue=live_request_queue,
            run_config=run_config,
        )
        logger.info(f"ADK run_live started for session {session_id}")

        # --- Bridge Tasks ---

        # Task: Listen to ADK events and send to WebSocket client
        async def adk_to_client():
            nonlocal session # Allow read/write of outer scope session
            logger.debug(f"[{session_id}] ADK -> Client task started.")
            audio_chunk_counter = 0 # Counter for audio chunks in this turn
            try:
                async for event in live_events:
                    # --- DEBUG: Log raw event structure ---
                    logger.debug(f"[{session_id}] Raw ADK Event Received: {event}")
                    # You might also try: logger.debug(f"[{session_id}] Raw ADK Event Dir: {dir(event)}")
                    # Or for more detail: logger.debug(f"[{session_id}] Raw ADK Event Vars: {vars(event)}")
                    # --- END DEBUG ---
                    # print(f"[{session_id}] ADK -> Client event: {event}")
                    message_to_send = None
                    event_processed = False # Flag to track if we sent something for this event
                    # print("Hello 1")

                    # --- ADDED INTERRUPTION CHECK ---
                    # Check if the event signals an interruption (attribute name might vary based on ADK Event structure)
                    # Based on Pastra example, interruption might be on a specific part of the event payload.
                    # Let's check if the event object itself has an 'interrupted' attribute for now.
                    if hasattr(event, 'interrupted') and event.interrupted:
                         logger.info(f"[{session_id}] Received interruption signal from ADK.")
                         # --- SEND INTERRUPTION MESSAGE TO CLIENT ---
                         await websocket.send_json({"type": "interrupted"})
                         logger.info(f"[{session_id}] Sent 'interrupted' signal to client.")
                         # --- END SEND INTERRUPTION MESSAGE ---
                         # Optionally break or return here if no further processing needed for interrupted events
                         # continue # Example: Skip further processing for this event
                    # --- END ADDED INTERRUPTION CHECK ---

                    # --- Process Different Event Payloads ---

                    # 1. Check for final textual response
                    # if event.is_final_response() and event.content and event.content.parts:
                    #      print("TEXT")
                    #      part = event.content.parts[0]
                    #      if part.text is not None: # Check for text part
                    #          message_to_send = {"type": "text", "data": part.text}
                    #          logger.info(f"[{session_id}] Sending final text to client: '{part.text[:50]}...'") # Log text start
                    #          await websocket.send_json(message_to_send)
                    #          event_processed = True
                         # Add elif blocks here if final response could be other types (e.g., final image)

                    # 2. Check for streaming audio
                    if event.content and event.content.parts:
                        #  print("AUDIO")
                         part = event.content.parts[0]
                         if part.inline_data and part.inline_data.mime_type.startswith("audio/"):
                              audio_bytes = part.inline_data.data
                              audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                              message_to_send = {"type": "audio", "data": audio_b64}

                              # --- ADDED AUDIO LOGGING ---
                              audio_chunk_counter += 1
                              if audio_chunk_counter == 1:
                                  logger.info(f"[{session_id}] Receiving audio stream from ADK...")
                              # Log periodically or just the first chunk to avoid flooding
                              logger.debug(f"[{session_id}] Sending audio chunk #{audio_chunk_counter} ({len(audio_bytes)} bytes) to client.")
                              # --- END ADDED LOGGING ---

                              await websocket.send_json(message_to_send)
                              event_processed = True
                    # else:
                    #     print("NOTHING")    

                    # 3. Check for turn completion signal (always send if present)
                    if event.turn_complete:
                        # --- ADDED AUDIO LOGGING ---
                        if audio_chunk_counter > 0:
                            logger.info(f"[{session_id}] Finished sending {audio_chunk_counter} audio chunks.")
                        audio_chunk_counter = 0 # Reset counter for next turn
                        # --- END ADDED LOGGING ---
                        await websocket.send_json({"type": "turn_complete"})
                        logger.info(f"[{session_id}] Sent turn_complete signal to client.")
                        event_processed = True # Mark as processed even if content was also sent

                    # Log skipped events
                    if not event_processed and not event.actions: # Avoid logging pure action events unless debugging actions specifically
                        # Log events that weren't audio/final text/turn complete
                        # Check if it's a tool call/response event before logging as "Skipping"
                        is_tool_event = bool(event.get_function_calls() or event.get_function_responses())
                        if not is_tool_event:
                             logger.debug(f"[{session_id}] Skipping event: Author={event.author}, Partial={event.partial}, Content Type={type(event.content.parts[0]).__name__ if event.content and event.content.parts else 'None'}")
                        else:
                            # Log tool calls/responses if needed for debugging the flow
                             if event.get_function_calls():
                                 logger.debug(f"[{session_id}] Processing event: Tool Call Requested by {event.author}")
                             elif event.get_function_responses():
                                 logger.debug(f"[{session_id}] Processing event: Tool Response from {event.author}")


                    # --- Apply State/Artifact Deltas ---
                    # Use the session_service to ensure proper handling (incl. persistence if needed)
                    if event.actions and (event.actions.state_delta or event.actions.artifact_delta):
                        logger.debug(f"[{session_id}] Appending event with actions: {event.actions}")
                        session_service.append_event(session, event)
                        # Reload session object to reflect committed changes locally
                        session = session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
                        logger.debug(f"[{session_id}] Session state possibly updated by event actions.")

            except WebSocketDisconnect:
                 logger.info(f"[{session_id}] WebSocket disconnected during ADK event processing.")
            except asyncio.CancelledError:
                 logger.info(f"[{session_id}] ADK -> Client task cancelled.")
            except Exception as e:
                 logger.error(f"[{session_id}] Error in ADK -> Client task: {e}", exc_info=True)
            finally:
                 logger.debug(f"[{session_id}] ADK -> Client task finished.")


        # Task: Listen to WebSocket client messages and send to ADK
        async def client_to_adk():
            nonlocal session # Allow read/write of outer scope session
            logger.debug(f"[{session_id}] Client -> ADK task started.")
            try:
                while True:
                    data = await websocket.receive_json()
                    message_type = data.get("type")
                    logger.debug(f"[{session_id}] Received '{message_type}' from client.")

                    if message_type == "audio":
                        audio_b64 = data.get("data", "")
                        if audio_b64:
                            audio_bytes = base64.b64decode(audio_b64)
                            if audio_bytes:
                                audio_blob = genai_types.Blob(mime_type='audio/pcm', data=audio_bytes)
                                # Use send_realtime for blobs
                                live_request_queue.send_realtime(blob=audio_blob)
                            else:
                                logger.warning(f"[{session_id}] Received empty audio data from client.")
                    elif message_type == "text":
                        text_data = data.get("data", "")
                        if text_data:
                            logger.info(f"[{session_id}] Sending text '{text_data}' to ADK.")
                            content = genai_types.Content(role='user', parts=[genai_types.Part(text=text_data)])
                            # Use send_content for structured Content
                            live_request_queue.send_content(content=content)
                    elif message_type == "end_of_turn":
                        logger.info(f"[{session_id}] Client indicated end of turn.")
                        # No direct method, let Gemini API infer turn end
                        logger.debug(f"[{session_id}] End of turn signal received, letting Gemini API infer turn end.")
                    elif message_type == "toggle_mock":
                         mock_value = data.get("value", False)
                         logger.info(f"[{session_id}] Setting mock_a2a_calls state to: {mock_value}")
                         # Create an event to signal the state change
                         state_update_event = Event(
                             author="ui_control", # Identify source
                             invocation_id = f"ui_mock_toggle_{uuid.uuid4().hex[:8]}",
                             actions=EventActions(
                                 state_delta={'mock_a2a_calls': mock_value}
                             )
                             # No content needed for this system event
                         )
                         # Use the service to append the event, which updates the state
                         session_service.append_event(session, state_update_event)
                         # Reload session object to see the change immediately in this context
                         session = session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
                         logger.info(f"[{session_id}] State 'mock_a2a_calls' updated via event to: {session.state.get('mock_a2a_calls')}")
                    else:
                         logger.warning(f"[{session_id}] Received unknown message type from client: {message_type}")

            except WebSocketDisconnect:
                logger.info(f"[{session_id}] Client WebSocket disconnected (detected in listener).")
                if live_request_queue: live_request_queue.close() # Signal ADK runner
            except asyncio.CancelledError:
                 logger.info(f"[{session_id}] Client -> ADK task cancelled.")
            except Exception as e_inner:
                logger.error(f"[{session_id}] Error in client listener task: {e_inner}", exc_info=True)
                if live_request_queue: live_request_queue.close() # Close queue on error
            finally:
                logger.debug(f"[{session_id}] Client -> ADK task finished.")


        # Run bridge tasks concurrently and wait for the first one to finish
        logger.info(f"[{session_id}] Starting ADK <-> WebSocket bridge tasks.")
        adk_run_task = asyncio.create_task(adk_to_client())
        client_listener_task = asyncio.create_task(client_to_adk())
        done, pending = await asyncio.wait(
            [adk_run_task, client_listener_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        logger.info(f"[{session_id}] One of the bridge tasks completed ({len(done)} done, {len(pending)} pending).")

        # Cancel any tasks that are still pending
        for task in pending:
            if not task.done():
                logger.debug(f"[{session_id}] Cancelling pending bridge task...")
                task.cancel()
                with suppress(asyncio.CancelledError): await task

    # --- Outer Exception Handling and Cleanup ---
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected gracefully for session: {session_id}")
    except asyncio.CancelledError:
         logger.info(f"WebSocket endpoint task cancelled for session: {session_id}")
    except Exception as e_outer:
        logger.error(f"Unhandled error in WebSocket endpoint for session {session_id}: {e_outer}", exc_info=True)
        # Attempt to notify client of the error if socket is still open-ish
        with suppress(Exception):
            await websocket.close(code=1011, reason=f"Internal Server Error: {type(e_outer).__name__}")
    finally:
        logger.info(f"Performing final cleanup for session: {session_id}")
        # Attempt to close the queue if it exists
        if live_request_queue:
             logger.debug(f"[{session_id}] Closing live request queue in final cleanup.")
             try:
                 live_request_queue.close()
             except Exception as q_close_err:
                 logger.warning(f"[{session_id}] Error closing LiveRequestQueue: {q_close_err}")
        # Cancel tasks again for safety
        if adk_run_task and not adk_run_task.done(): adk_run_task.cancel()
        if client_listener_task and not client_listener_task.done(): client_listener_task.cancel()
        # Optional: Remove session from memory on disconnect
        if session:
            try:
                session_service.delete_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
                logger.info(f"Session removed from service: {session_id}")
            except Exception as del_err:
                 logger.warning(f"[{session_id}] Error deleting session: {del_err}")
        logger.info(f"WebSocket connection fully closed for session: {session_id}")


# --- Static File Serving ---
if not STATIC_DIR.is_dir():
    logger.error(f"Static directory not found at {STATIC_DIR}. UI will not be served.")
else:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"Serving static files from {STATIC_DIR}")

    @app.get("/")
    async def read_index():
        """Serves the main index.html file."""
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            logger.error("index.html not found in static directory.")
            return {"error": "index.html not found"}, 404
        logger.debug("Serving index.html")
        return FileResponse(str(index_path))

# --- Server Startup (for running directly) ---
if __name__ == "__main__":
    # Ensure project root is in path if needed
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.debug(f"Added project root to sys.path for direct execution: {project_root}")

    LIVE_SERVER_HOST = os.getenv("LIVE_SERVER_HOST", "127.0.0.1")
    LIVE_SERVER_PORT = int(os.getenv("LIVE_SERVER_PORT", "8082")) # Defaulting back to 8081
    logger.info(f"Starting Project Horizon Live Server on http://{LIVE_SERVER_HOST}:{LIVE_SERVER_PORT}")
    try:
        import uvicorn
        # Use string format for app location as recommended by uvicorn
        uvicorn.run("app.live_server:app", host=LIVE_SERVER_HOST, port=LIVE_SERVER_PORT, log_level="info") # Set uvicorn log level
    except ImportError:
         logger.critical("Uvicorn not installed. Run 'pip install uvicorn'.")
    except Exception as startup_error:
         logger.critical(f"Failed to start Uvicorn server: {startup_error}", exc_info=True)