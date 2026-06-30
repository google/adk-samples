"""Standalone health check proxy for bidi-demo Cloud Monitoring uptime checks.

Connects to the bidi-demo WebSocket endpoint as a client and verifies
the model responds to a text query. Designed to run as a separate Cloud Run
service that monitors the main bidi-demo service end-to-end:

    Cloud Monitoring -> GET /health/live -> this app -> WS -> bidi-demo -> Live API

Endpoints:
    GET /health      - Basic reachability check (returns {"status": "ok"})
    GET /health/live - End-to-end model check via bidi-demo WebSocket

Environment variables:
    BIDI_DEMO_URL: WebSocket base URL of the bidi-demo service
                   (default: ws://localhost:8000)

Local testing:
    # Start bidi-demo on port 8000, then:
    BIDI_DEMO_URL=ws://localhost:8000 \
    uv run --project .. uvicorn health:app --host 0.0.0.0 --port 8001

    curl http://localhost:8001/health/live

Deploy to Cloud Run:
    gcloud run deploy bidi-demo-health \
      --source . \
      --project "${GOOGLE_CLOUD_PROJECT}" \
      --region "${GOOGLE_CLOUD_LOCATION}" \
      --allow-unauthenticated \
      --timeout 30 \
      --min-instances 0 \
      --max-instances 1 \
      --command='sh' \
      --args='-c,uvicorn health:app --host 0.0.0.0 --port $PORT' \
      --set-env-vars BIDI_DEMO_URL=wss://bidi-demo-xxx.us-east1.run.app

Create uptime check:
    gcloud monitoring uptime create "bidi-demo /health/live" \
      --resource-type=uptime-url \
      --resource-labels=host=bidi-demo-health-xxx.us-east1.run.app,project_id=PROJECT \
      --protocol=https \
      --path=/health/live \
      --port=443 \
      --request-method=get \
      --matcher-content=Tokyo \
      --period=5 \
      --timeout=30 \
      --regions=europe,asia-pacific,usa-iowa \
      --validate-ssl=true \
      --project=PROJECT
"""

import asyncio
import json
import os
import uuid

import websockets
from fastapi import FastAPI
from fastapi.responses import JSONResponse

BIDI_DEMO_URL = os.getenv("BIDI_DEMO_URL", "ws://localhost:8000")
HEALTH_CHECK_TIMEOUT = 20
HEALTH_CHECK_QUERY = "What time is it in Tokyo?"

app = FastAPI()


@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "ok"}


@app.get("/health/live")
async def health_live():
    """End-to-end health check that connects to bidi-demo as a WebSocket client."""
    user_id = "uptime-check"
    session_id = f"health-{uuid.uuid4().hex[:12]}"
    ws_url = f"{BIDI_DEMO_URL}/ws/{user_id}/{session_id}"

    transcript_parts = []

    async def _check():
        async with websockets.connect(ws_url) as ws:
            # Send text query
            await ws.send(
                json.dumps(
                    {
                        "type": "text",
                        "text": HEALTH_CHECK_QUERY,
                    }
                )
            )

            # Collect events until turn_complete
            async for message in ws:
                event = json.loads(message)

                # Collect text from content parts
                content = event.get("content")
                if content and content.get("parts"):
                    for part in content["parts"]:
                        if part.get("text"):
                            transcript_parts.append(part["text"])

                # Stop after model completes its turn
                if event.get("turnComplete"):
                    break

    try:
        await asyncio.wait_for(_check(), timeout=HEALTH_CHECK_TIMEOUT)
        transcript = "".join(transcript_parts)
        if not transcript:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "error": "No transcript received"},
            )
        return {"status": "ok", "transcript": transcript}
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": "Model response timed out"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": str(e)},
        )
