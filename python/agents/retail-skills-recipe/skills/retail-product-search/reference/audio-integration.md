# Product Search - Audio Integration Reference

Voice-based product search using the Gemini Live API via the `google-genai` SDK.

---

## Overview

This reference documents the voice integration for the product search agent.
The implementation uses the `google-genai` SDK's Live API for bidirectional
audio streaming with tool calling, replacing the previous raw WebSocket approach.

**Script**: `scripts/live_search.py`

---

## Architecture

```
User speaks  -->  Gemini Live API (google-genai SDK)  -->  Tool call: search_products
    ^                      |                                       |
    |                      v                                       v
Speaker  <--  Audio response  <--  Model formats results  <--  Vector Search 2.0
```

The Live API handles:
- Speech-to-text (automatic, via `input_audio_transcription`)
- Intent understanding and tool invocation
- Text-to-speech for the response (automatic, via `response_modalities: ["AUDIO"]`)

No separate intent extractor, TTS formatter, or session manager is needed --
the Live API manages all of this natively.

---

## Quick Start

### Text mode (no microphone needed)

```bash
python scripts/live_search.py \
    --project-id YOUR_PROJECT \
    --collection projects/PROJECT/locations/REGION/collections/COLLECTION \
    --mode text
```

### Voice mode

```bash
# Install voice dependencies
pip install pyaudio

# macOS may need portaudio first:
# brew install portaudio

python scripts/live_search.py \
    --project-id YOUR_PROJECT \
    --collection projects/PROJECT/locations/REGION/collections/COLLECTION \
    --mode voice
```

### Using design-spec.md for defaults

```bash
python scripts/live_search.py --config design-spec.md --mode voice
```

---

## How It Works

### 1. Session Setup

The script connects to Gemini Live API with:
- A system instruction that defines the assistant's behavior
- A `search_products` tool declaration for catalog queries
- Audio transcription enabled (both input and output)
- Voice configuration (default: Kore)

```python
from google import genai
from google.genai import types

client = genai.Client()

config = {
    "system_instruction": SYSTEM_INSTRUCTION,
    "response_modalities": ["AUDIO"],
    "tools": [SEARCH_TOOL],
    "input_audio_transcription": {},
    "output_audio_transcription": {},
    "speech_config": {
        "voice_config": {
            "prebuilt_voice_config": {"voice_name": "Kore"}
        }
    },
}

async with client.aio.live.connect(model=MODEL, config=config) as session:
    # Session is ready for audio streaming
```

### 2. Audio Streaming (Voice Mode)

Three concurrent tasks run in parallel:

1. **capture_audio**: Reads microphone at 16kHz mono PCM, sends to Gemini
2. **play_audio**: Plays audio responses at 24kHz through speakers
3. **receive_responses**: Processes tool calls, transcriptions, and audio data

```python
# Sending audio input
await session.send_realtime_input(
    audio=types.Blob(
        data=audio_chunk,
        mime_type="audio/pcm;rate=16000",
    )
)

# Receiving audio output
async for response in session.receive():
    if response.server_content and response.server_content.model_turn:
        for part in response.server_content.model_turn.parts:
            if part.inline_data:
                # Queue audio for playback
                await audio_out_queue.put(part.inline_data.data)
```

### 3. Tool Calling

When the model decides to search, it invokes the `search_products` tool.
The script executes the search against Vector Search 2.0 and returns results:

```python
# Tool declaration
SEARCH_TOOL = {
    "function_declarations": [{
        "name": "search_products",
        "description": "Search the product catalog using a natural language query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default 5)",
                },
            },
            "required": ["query"],
        },
    }]
}

# Handling tool calls
if response.tool_call:
    for fc in response.tool_call.function_calls:
        result = search_products_in_collection(
            query=fc.args["query"],
            collection_path=collection_path,
        )
        await session.send_tool_response(
            function_responses=[
                types.FunctionResponse(
                    id=fc.id,
                    name=fc.name,
                    response={"result": result},
                )
            ]
        )
```

### 4. Transcriptions

Both input (what the user said) and output (what the model said) are
transcribed automatically:

```python
if sc.input_transcription and sc.input_transcription.text:
    print(f"  You: {sc.input_transcription.text}")

if sc.output_transcription and sc.output_transcription.text:
    print(f"  Assistant: {sc.output_transcription.text}")
```

---

## Configuration

### Voice Options

Available voices: `Kore`, `Aoede`, `Charon`, `Fenrir`, `Puck`, `Leda`

```bash
python scripts/live_search.py --voice Aoede --mode voice
```

### Audio Format

- **Input**: 16kHz, 16-bit PCM, mono (`audio/pcm;rate=16000`)
- **Output**: 24kHz, 16-bit PCM, mono

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID (fallback for `--project-id`) |
| `VECTOR_SEARCH_COLLECTION` | Collection resource name (fallback for `--collection`) |
| `GOOGLE_GENAI_USE_VERTEXAI` | Set to `True` for Vertex AI (set automatically) |

---

## Dependencies

Core (always required):
```
google-genai>=1.0
google-cloud-vectorsearch>=0.1
pyyaml>=6.0
```

Voice mode (optional):
```
pyaudio>=0.2.14
```

macOS setup for pyaudio:
```bash
brew install portaudio
pip install pyaudio
```

---

## Text Mode vs Voice Mode

| Feature | Text Mode | Voice Mode |
|---------|-----------|------------|
| Input | Keyboard | Microphone |
| Output | Text in terminal | Audio through speakers + transcript |
| Dependencies | google-genai only | google-genai + pyaudio |
| Use case | Testing, CI, demos | Production, user-facing |
| Tool calling | Same | Same |

Text mode uses `send_client_content` with text turns.
Voice mode uses `send_realtime_input` with PCM audio blobs.

---

## Troubleshooting

### "No module named 'pyaudio'"
Install pyaudio: `pip install pyaudio`. On macOS, install portaudio first:
`brew install portaudio`.

### "No product collection configured"
Set `--collection` or `VECTOR_SEARCH_COLLECTION` env var to your
Vector Search 2.0 collection resource name.

### Audio is choppy
- Ensure stable network connection (Live API uses persistent streaming)
- Close other audio applications
- Try a lower-latency audio device

### Tool calls not triggering
- Verify the collection has products ingested
- Check that the system instruction matches your product domain
- Try being more specific in your query

---

## Extending

### Adding more tools

Add tool declarations to the `SEARCH_TOOL` dict and handle them in
`_handle_tool_call()`. Examples:

- `get_product_details`: Fetch full details for a specific product ID
- `add_to_cart`: Add a product to the shopping cart
- `check_availability`: Check stock at a specific store

### Connecting to an existing agent

The `search_products_in_collection()` function can be replaced with a call
to your existing agent's retrieval logic. The Live API session just needs
a string result from each tool call.
