"""Bounded Interactions API prototype for Phase 14b step 3.

Goal: establish, with the SDK's TYPED request path (raw dicts proven broken,
ws01), (a) the working request/response shape for text_to_video and
image_to_video on gemini-omni-flash-preview, (b) polling mechanics via
client.interactions.get(), and (c) what previous_interaction_id actually
needs for a meaningful edit. Read-only against the API except for the
interactions it creates. Every attempt prints its full outcome; nothing is
asserted — findings go to omni-prototype-findings.md.

Discoveries baked in from probe round 1 (log: /tmp/omni-probe.log):
- SDK 2.14.0: create() is `create(*, request=None, ..., **body)` — `model=`,
  `input=`, `previous_interaction_id=` etc. pass through **body; get() takes
  positional `id` (NOT `interaction_id=`); typed surface lives in
  `google.genai.interactions` (google.genai.types only has ReplayInteraction);
  results are on `Interaction.output_video` / `.steps`, not `.outputs`.
- Bare `create(model=..., input="<text>")` on gemini-omni-flash-preview
  SYNCHRONOUSLY returns status='completed' with inline video/mp4 bytes.
- `response_modalities=["video"]` + `background=True` together produced an
  async `video-*` interaction that immediately failed (code 3 invalid
  argument) — round 2 isolates which parameter is the poison.
- List input in turn format ({"role": ..., "content": ...}) is rejected:
  "use step_list input format instead of turn_list" — round 2 uses
  {"type": "user_input", "content": [...]} steps.

Run: set -a; source app/.env; set +a
     GOOGLE_CLOUD_LOCATION=global .venv/bin/python scripts/probe_omni_interactions.py
"""

import base64
import inspect
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai
from google.genai import types

MODEL = os.getenv("MODEL_NAME_GENERATED_3")
TEXT_PROMPT = (
    "A 4-second product hero shot of a matte black cold brew can rotating "
    "slowly on a marble counter, soft studio light."
)
FASHION_PNG = (
    Path(__file__).resolve().parent.parent
    / "generated"
    / "model-comparison"
    / os.getenv("MODEL_NAME_GENERATED_2")
    / "fashion-scene.png"
)
OUT_DIR = Path("/tmp/omni-probe-outputs")  # noqa: S108


def show(label, fn):
    print(f"\n=== {label} ===")
    try:
        out = fn()
        print(f"OK: {out!r}"[:1200])
        return out
    except Exception as e:
        print(f"FAIL [{type(e).__name__}]: {e}"[:1200])
        return None


def summarize(interaction):
    """Compact, informative dump of an Interaction's interesting fields."""
    if interaction is None:
        return None
    steps = getattr(interaction, "steps", None) or []
    step_summary = []
    for s in steps:
        content = getattr(s, "content", None) or []
        entry = {
            "type": type(s).__name__,
            "content_types": [type(c).__name__ for c in content]
            if isinstance(content, list)
            else type(content).__name__,
        }
        error = getattr(s, "error", None)
        if error is not None:
            entry["error"] = repr(error)
        step_summary.append(entry)
    video = getattr(interaction, "output_video", None)
    usage = getattr(interaction, "usage", None)
    return {
        "id": getattr(interaction, "id", None),
        "status": getattr(interaction, "status", None),
        "model": getattr(interaction, "model", None),
        "previous_interaction_id": getattr(
            interaction, "previous_interaction_id", None
        ),
        "steps": step_summary,
        "output_text": (getattr(interaction, "output_text", None) or "")[:200],
        "output_video": {
            "mime_type": getattr(video, "mime_type", None),
            "uri": getattr(video, "uri", None),
            "data_bytes": len(getattr(video, "data", b"") or b""),
        }
        if video is not None
        else None,
        "total_output_tokens": getattr(usage, "total_output_tokens", None),
    }


def save_video(interaction, name):
    video = getattr(interaction, "output_video", None) if interaction else None
    if video is None:
        return "no output_video"
    data = getattr(video, "data", None)
    if data:
        raw = base64.b64decode(data) if isinstance(data, str) else data
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{name}.mp4"
        path.write_bytes(raw)
        return f"saved {len(raw)} bytes -> {path}"
    return f"no inline data (uri={getattr(video, 'uri', None)})"


def poll(client, interaction_id, budget_s=600, interval_s=10):
    waited = 0
    latest = client.interactions.get(interaction_id, timeout=120.0)
    while getattr(latest, "status", None) in (
        None,
        "in_progress",
        "processing",
        "queued",
    ):
        if waited >= budget_s:
            raise TimeoutError(f"interaction poll timed out after {budget_s}s")
        time.sleep(interval_s)
        waited += interval_s
        latest = client.interactions.get(interaction_id, timeout=120.0)
        print(f"  ... {waited}s status={getattr(latest, 'status', None)}")
    print(f"  terminal after {waited}s: {getattr(latest, 'status', None)}")
    return latest


def main() -> int:
    client = genai.Client()

    # 0. Discover the typed surface before calling anything.
    print("interactions.create signature:")
    print(inspect.signature(client.interactions.create))
    print("\ninteractions.get signature:")
    print(inspect.signature(client.interactions.get))
    typed = [n for n in dir(types) if "interaction" in n.lower()]
    print(f"\ntypes.*Interaction*: {typed}")
    from google.genai import interactions as interactions_types

    print(
        "google.genai.interactions video/status types:",
        [
            n
            for n in dir(interactions_types)
            if "video" in n.lower() or n == "InteractionStatus"
        ],
    )

    # 1. text_to_video, minimal working shape (round 1: completes SYNCHRONOUSLY
    #    with inline mp4 bytes — no modalities/background flags needed).
    t0 = time.monotonic()
    plain = show(
        "create: text_to_video, bare text input (known-good round-1 shape)",
        lambda: client.interactions.create(
            model=MODEL, input=TEXT_PROMPT, timeout=600.0
        ),
    )
    print(f"latency: {round(time.monotonic() - t0, 1)}s")
    print(f"summary: {summarize(plain)}")
    show(
        "save text_to_video output", lambda: save_video(plain, "text_to_video")
    )
    if plain is None:
        print("\ntext_to_video creation failed — record and stop.")
        return 1

    # 2. Retrieval/polling mechanics: fetch the same interaction by id.
    fetched = show(
        "interactions.get on completed id", lambda: poll(client, plain.id)
    )
    print(f"summary: {summarize(fetched)}")

    # 3. Isolate round-1 failure: background=True ALONE (no response_modalities).
    bg = show(
        "create: background=True alone",
        lambda: client.interactions.create(
            model=MODEL, input=TEXT_PROMPT, background=True, timeout=600.0
        ),
    )
    print(f"summary: {summarize(bg)}")
    if bg is not None and getattr(bg, "status", None) not in (
        "completed",
        "failed",
    ):
        bg_final = show(
            "poll background interaction", lambda: poll(client, bg.id)
        )
        print(f"summary: {summarize(bg_final)}")
        show(
            "save background output",
            lambda: save_video(bg_final, "text_to_video_bg"),
        )

    # 4. Isolate round-1 failure: response_modalities=["video"] ALONE (sync).
    rm = show(
        'create: response_modalities=["video"] alone',
        lambda: client.interactions.create(
            model=MODEL,
            input=TEXT_PROMPT,
            response_modalities=["video"],
            timeout=600.0,
        ),
    )
    print(f"summary: {summarize(rm)}")

    # 5. image_to_video: step_list input format (turn_list rejected, round 1),
    #    using Task 1's fashion scene PNG.
    if FASHION_PNG.exists():
        image_b64 = base64.b64encode(FASHION_PNG.read_bytes()).decode("ascii")
        i2v = show(
            "create: image_to_video (step_list input: user_input step)",
            lambda: client.interactions.create(
                model=MODEL,
                input=[
                    {
                        "type": "user_input",
                        "content": [
                            {
                                "type": "image",
                                "data": image_b64,
                                "mime_type": "image/png",
                            },
                            {
                                "type": "text",
                                "text": "Animate this exact scene into a 4-second "
                                "video: gentle camera push-in, fabric moving in a "
                                "light breeze, golden-hour glow.",
                            },
                        ],
                    }
                ],
                timeout=600.0,
            ),
        )
        print(f"summary: {summarize(i2v)}")
        show(
            "save image_to_video output",
            lambda: save_video(i2v, "image_to_video"),
        )
    else:
        print(f"\nNOTE: {FASHION_PNG} missing — image_to_video attempt skipped")

    # 6. Edit via previous_interaction_id: text delta alone, same minimal shape
    #    as the known-good create (round-1 edit failure was confounded with
    #    response_modalities+background; this isolates the lineage mechanism).
    edit = show(
        "edit via previous_interaction_id (text delta only)",
        lambda: client.interactions.create(
            model=MODEL,
            input="Same video, but make the lighting warm golden hour.",
            previous_interaction_id=plain.id,
            timeout=600.0,
        ),
    )
    print(f"summary: {summarize(edit)}")
    show("save edit output", lambda: save_video(edit, "edit_text_delta"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
