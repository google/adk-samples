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

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Add the base directory of the project to sys.path to make 'utils' importable
base_dir = Path(__file__).parent.parent.resolve()
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

load_dotenv(base_dir / "info_gather_agent" / ".env")

from utils.audio_utils import (  # noqa: E402
    concatenate_and_resample_audio,
    find_sorted_output_audio_files,
)
from utils.config import TTS_EVAL_GENERATE_CONFIG, TTS_EVAL_MODEL  # noqa: E402

# =====================================================================
# Main Execution Flow
# =====================================================================


def main():
    eval_dir = Path(__file__).parent.resolve()
    base_dir = eval_dir.parent
    artifacts_dir = base_dir / "artifacts"

    if not artifacts_dir.exists():
        print(f"Error: artifacts directory not found at: {artifacts_dir}")
        return

    # Step 1 & 2: Scan and load sorted audio files from output subdirectories
    audio_files = find_sorted_output_audio_files(artifacts_dir)
    if not audio_files:
        print(
            "No raw audio files found inside the output directories. Speak to the agent first!"
        )
        return

    print(
        f"Found {len(audio_files)} raw audio segments. Concatenating and downsampling..."
    )

    # Step 3: Concatenate and downsample each chunk to 16 kHz
    combined_wav_path = eval_dir / "tts_combined_output.wav"
    success = concatenate_and_resample_audio(
        audio_files=audio_files,
        output_wav_path=combined_wav_path,
        orig_sr=24000,
        target_sr=16000,
    )
    if not success:
        return

    # Step 4: Initialize Gemini Client and Call Model
    print("\nInitializing Gemini Client for Speech Quality Assessment...")
    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION"),
    )

    with open(combined_wav_path, "rb") as audio_f:
        audio_bytes = audio_f.read()

    response = None
    try:
        print(f"Attempting to call model: {TTS_EVAL_MODEL}...")
        response = client.models.generate_content(
            model=TTS_EVAL_MODEL,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                "Please perform the speech quality evaluation on this audio.",
            ],
            config=TTS_EVAL_GENERATE_CONFIG,
        )
    except Exception as e:
        print(f"Model {TTS_EVAL_MODEL} invocation failed: {e}")

    if not response:
        print(
            "\nError: All Gemini models failed to evaluate the audio. Please verify your Google Cloud Vertex AI quotas and settings."
        )
        return

    # Print Evaluation Report
    try:
        report = json.loads(response.text)
        print("\n" + "=" * 50)
        print("TTS QUALITY EVALUATION REPORT")
        print("=" * 50)
        print(f"Mean Opinion Score (MOS): {report.get('mos_score')}")
        print(f"Rationale:\n{report.get('rationale')}")
        print("=" * 50)
    except Exception:
        print("\nFailed to parse report JSON response. Raw response:")
        print(response.text)


if __name__ == "__main__":
    main()
