import os
import wave
import re
import json
from pathlib import Path
import numpy as np


def resample_audio(data, orig_sr, target_sr):
    """Resamples audio data using linear interpolation."""
    if orig_sr == target_sr:
        return data

    duration = len(data) / orig_sr
    num_target_samples = int(duration * target_sr)

    orig_indices = np.arange(len(data))
    target_indices = np.linspace(0, len(data) - 1, num_target_samples)

    return np.interp(target_indices, orig_indices, data).astype(np.int16)


def extract_timestamp(filename):
    """Extracts the millisecond timestamp from the ADK artifact filename."""
    match = re.search(r"_(\d{13})\.", filename)
    if match:
        return int(match.group(1))
    match = re.search(r"_(\d{10,})\.", filename)
    if match:
        return int(match.group(1))
    return 0


def concatenate_and_resample_audio(audio_files, output_wav_path, orig_sr=24000, target_sr=16000):
    """Concatenates a list of raw PCM audio files, resamples them, and saves as a WAV file."""
    all_resampled = []

    for path in audio_files:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.int16)
        
        if len(data) == 0:
            continue
            
        resampled = resample_audio(data, orig_sr, target_sr)
        all_resampled.append(resampled)

    if not all_resampled:
        print("No valid audio data found in the segments.")
        return False

    merged_data = np.concatenate(all_resampled)
    output_path = Path(output_wav_path).resolve()

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(target_sr)
        wav_file.writeframes(merged_data.tobytes())

    print(f"Pruned, merged and downsampled audio saved to: {output_path}")
    return True



def _find_session_dir(pcm_file: Path, artifacts_root: Path) -> Path:
    """Finds the parent session directory for a given pcm file."""
    current = pcm_file.parent
    while current != artifacts_root:
        if current.parent.name == "sessions":
            return current
        current = current.parent
    return None


def find_sorted_output_audio_files(artifacts_dir: str) -> list:
    """Scans the artifacts directory for output audio subdirectories and returns all sorted audio paths."""
    artifacts_root = Path(artifacts_dir)
    if not artifacts_root.exists():
        return []

    output_subdirs = []
    for root, dirs, _ in os.walk(artifacts_root):
        for d in dirs:
            if "output" in d.lower():
                output_subdirs.append(Path(root) / d)

    audio_files = []
    for subdir in output_subdirs:
        for root, _, files in os.walk(subdir):
            for f in files:
                f_path = Path(root) / f
                if f_path.is_file() and f_path.suffix != ".json":
                    audio_files.append(f_path)

    # Sort by timestamp to guarantee chronological order of conversation
    audio_files.sort(key=lambda x: extract_timestamp(x.name))
    return audio_files


def convert_artifacts_to_wav(
    artifacts_dir: str = "./artifacts",
    output_filename: str = None,
    session_id: str = None
):
    """Scans artifacts_dir for PCM audio files, merges, resamples, and saves them to output_filename."""
    if output_filename is None:
        if session_id:
            output_filename = f"./{session_id}.wav"
        else:
            output_filename = "./session.wav"

    artifacts_root = Path(artifacts_dir)
    
    # Smart Fallbacks for ADK Web Framework directories
    if not artifacts_root.exists() or not any(artifacts_root.rglob("*.pcm")):
        fallbacks = [
            Path("./.adk/artifacts"),
            Path("./info_gather_agent/.adk/artifacts"),
            Path("../.adk/artifacts")
        ]
        for fallback in fallbacks:
            if fallback.exists() and any(fallback.rglob("*")):
                artifacts_root = fallback
                print(f"Found active artifacts directory at: {artifacts_root.resolve()}")
                break

    if not artifacts_root.exists():
        print(f"Artifacts directory not found. Please speak to the agent via the Web UI first to generate audio blobs!")
        return False

    sessions = {}
    for pcm_file in artifacts_root.rglob("*"):
        if (
            not pcm_file.is_file()
            or pcm_file.name == "metadata.json"
            or pcm_file.suffix == ".json"
        ):
            continue

        if "adk_live_audio_storage_" not in pcm_file.name:
            continue

        session_dir = _find_session_dir(pcm_file, artifacts_root)


        if not session_dir:
            continue

        session_id = session_dir.name
        if session_id not in sessions:
            sessions[session_id] = []

        sample_rate = 16000
        if "output_audio" in pcm_file.name:
            sample_rate = 24000
        elif "rate=16000" in pcm_file.name:
            sample_rate = 16000

        timestamp = extract_timestamp(pcm_file.name)
        sessions[session_id].append((timestamp, pcm_file, sample_rate))

    target_sr = 16000
    all_resampled_audio = []

    # Merge all sessions and all chunks found
    for sid, artifacts in sessions.items():
        if session_id and sid != session_id:
            continue
        if not artifacts:
            continue

        # Sort by timestamp to preserve conversation order
        artifacts.sort(key=lambda x: x[0])

        for ts, path, sr in artifacts:
            with open(path, "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.int16)

            if len(data) == 0:
                continue

            resampled = resample_audio(data, sr, target_sr)
            all_resampled_audio.append(resampled)

    if not all_resampled_audio:
        print("No audio artifacts found to convert.")
        return False

    merged_data = np.concatenate(all_resampled_audio)
    output_path = Path(output_filename).resolve()

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(target_sr)
        wav_file.writeframes(merged_data.tobytes())

    print(f"Success! Converted conversation saved to: {output_path}")
    return True
