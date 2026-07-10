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

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.genai import types
from pydantic import BaseModel, Field
from utils.prompt import TTS_EVAL_SYSTEM_INSTRUCTION

# Active model configuration definitions

# Model name for the real-time live conversational agent Alex
LIVE_AGENT_MODEL = "gemini-live-2.5-flash-native-audio"

# Model name for the Text-to-Speech (TTS) quality evaluation script
TTS_EVAL_MODEL = "gemini-2.5-pro"

# Centralized RunConfig for live ADK bidirectional audio session
LIVE_AGENT_RUN_CONFIG = RunConfig(
    realtime_input_config=types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(
            disabled=False,
            start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
            end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
            prefix_padding_ms=20,
            silence_duration_ms=150,
        )
    ),
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                voice_name="Achird",
            )
        ),
        language_code="en-US",
    ),
    session_resumption=types.SessionResumptionConfig(transparent=True),
    streaming_mode=StreamingMode.BIDI,
    response_modalities=["AUDIO"],
    input_audio_transcription=types.AudioTranscriptionConfig(),
    output_audio_transcription=types.AudioTranscriptionConfig(),
    save_live_blob=True,
)


# Structured JSON Output Schema for TTS Evaluations
class TtsEvaluation(BaseModel):
    mos_score: int = Field(
        ...,
        description="Overall speech quality Mean Opinion Score (MOS) from 1 to 5.",
    )
    rationale: str = Field(
        ...,
        description=(
            "Detailed speech quality analysis. If the mos_score is 3 or below, "
            "you MUST explicitly detail what went wrong (accent shift, robotic tones, "
            "distortions, etc.), including the specific part of the speech/phrase where "
            "the issue occurred and what was being spoken at that moment."
        ),
    )


# Model configuration for the Text-to-Speech (TTS) quality evaluation call

TTS_EVAL_GENERATE_CONFIG = types.GenerateContentConfig(
    system_instruction=TTS_EVAL_SYSTEM_INSTRUCTION,
    response_mime_type="application/json",
    response_schema=TtsEvaluation,
    seed=123,
)
